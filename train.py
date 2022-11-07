import time
import argparse
import numpy as np
import random
import shortuuid
import copy

import pandas as pd
import torch
import torch.nn.functional as F
import torch.optim as optim
import tqdm
from torch.utils.data import DataLoader

from sources.model import BeliefAutoencoder
from sources.preprocessing import preprocessing


def set_seed(seed=20190827):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def get_all_results(hyperparameters: dict, current_epoch: int, train_loss: float, kld_loss: float, recon_loss: float):
    result = {}
    for name, values in hyperparameters.items():
        result[name] = values
    result['current_epoch'] = current_epoch
    result['train_loss'] = train_loss
    result['kld'] = kld_loss
    result['recon_loss'] = recon_loss
    return result


def validate(dataloader, model, temp):
    train_loss = 0.
    recon_loss = 0.
    kld_loss = 0.
    with torch.no_grad():
        model.eval()
        for batch in dataloader:
            x, y, seq_lens = batch
            y_hat, qy = model(x, temperature=temp)
            loss, recon, kld = model.loss_function(y=y, y_hat=y_hat, qy=qy)
            train_loss += loss.item()
            recon_loss += recon.item()
            kld_loss += kld.item()
    return train_loss / len(dataloader), recon_loss / len(dataloader), kld_loss / len(dataloader)


def train(train_id: str, emb_dim: int, h_dim: int, latent_dim: int, categorical_dim: int = 2, batch_size: int = 128,
          save_model: bool = False, initial_temp: float = 1., min_temp: float = 0.5, epochs: int = 100,
          anneal_rate: float = 0.00003, activation: str = 'gumbel', model_name: str = None, lr: float = 1e-3,
          dropout_rate: float = 0.5):
    hyperparameters = locals()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    preprocessed_data = preprocessing("data/all_sentences_join.csv", device)
    vocab = preprocessed_data.vocab
    dataset = preprocessed_data.dataset
    model = BeliefAutoencoder(emb_dim=emb_dim, h_dim=h_dim, vocab=vocab, latent_dim=latent_dim,
                              categorical_dim=categorical_dim, device=device, activation=activation,
                              dropout_rate=dropout_rate)
    model.to(device)
    train_dataloader = DataLoader(dataset, batch_size=batch_size)
    optimizer = optim.Adam(model.parameters(), lr=lr)  # , betas=(0.5, 0.999))

    temp = initial_temp
    results = []
    best_loss = 999
    best_epoch = -1
    best_state = None
    for epoch in range(epochs):
        model.train()
        for batch_idx, batch in tqdm.tqdm(enumerate(train_dataloader), total=len(train_dataloader)):
            x, y, seq_lens = batch
            optimizer.zero_grad()
            y_hat, qy = model(x, temperature=temp)
            loss, recon, kld = model.loss_function(y=y, y_hat=y_hat, qy=qy)
            loss.backward()
            optimizer.step()
            if batch_idx % 100 == 1:
                temp = np.maximum(temp * np.exp(-anneal_rate * batch_idx), min_temp)

        train_loss, recon_loss, kld_loss = validate(train_dataloader, model, temp)
        if best_loss > train_loss:
            best_state = copy.deepcopy(model.state_dict())
            best_epoch = epoch
            best_loss = train_loss

        print(
            f"Epoch {epoch} - Train loss {train_loss:.4f} - Temp {temp:.4f} - recon_loss {recon_loss:.4f} - kld {kld_loss:.4f}")
        epoch_result = get_all_results(hyperparameters=hyperparameters,
                                       current_epoch=epoch,
                                       train_loss=train_loss,
                                       kld_loss=kld_loss,
                                       recon_loss=recon_loss)
        results.append(epoch_result)

    model.load_state_dict(best_state)
    model.eval()

    # train_loss, recon_loss, kld_loss = validate(train_dataloader, model, temp)
    # print(f"Best train_loss = {train_loss:.4f} - best epoch {best_epoch}")
    print(f"Best epoch {best_epoch}")

    if save_model:
        if model_name is not None:
            import os
            model_path = f"models/{model_name}"
            os.makedirs(model_path, exist_ok=True)
        else:
            model_path = f"models/"

        rep_dim = latent_dim * categorical_dim
        torch.save(best_state,
                   f"{model_path}/belief-autoencoder-{activation}-{emb_dim}-{h_dim}-{rep_dim}-{train_id}.pth")

    return pd.DataFrame(results)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--emb_dim", type=int, default=64, help="Dimension of embedding layer")
    parser.add_argument("--batch_size", type=int, default=100, help="Batch Size")
    parser.add_argument("--h_dim", type=int, default=128, help="RNN hidden dim")
    parser.add_argument("--epochs", type=int, default=100, help="number of epochs to train")
    parser.add_argument("--anneal_rate", type=float, default=0.00003, help="Gumbel anneal rate")
    parser.add_argument("--initial_temp", type=int, default=1, help="Gumbel initial temperature")
    parser.add_argument("--min_temp", type=float, default=0.5, help="Gumbel min temperature")
    parser.add_argument("--latent_dim", type=int, default=16, help="Dimension of latent vector")
    parser.add_argument("--categorical_dim", type=int, default=2, help="Number of categories")
    parser.add_argument("--save_model", action='store_true', default=False)
    parser.add_argument("--activation", type=str, default='gumbel')
    parser.add_argument("--model_name", type=str)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--dropout_rate", type=float, default=0.2)

    args = parser.parse_args()

    train_id = shortuuid.ShortUUID().random(length=8)

    print(f"Start training {train_id} - Args {args}")

    set_seed()

    df_results = train(train_id=train_id,
                       emb_dim=args.emb_dim,
                       h_dim=args.h_dim,
                       batch_size=args.batch_size,
                       epochs=args.epochs,
                       save_model=args.save_model,
                       initial_temp=args.initial_temp,
                       min_temp=args.min_temp,
                       latent_dim=args.latent_dim,
                       categorical_dim=args.categorical_dim,
                       anneal_rate=args.anneal_rate,
                       activation=args.activation,
                       model_name=args.model_name,
                       lr=args.lr,
                       dropout_rate=args.dropout_rate)

    df_results.to_csv(f'train_results/results_{train_id}.csv', index=False)
