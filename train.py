import time
import argparse
import numpy as np
import random

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


#
def loss_function(y, y_hat, qy, categorical_dim, vocab_size, eps=1e-20):
    batch_size = y.size(0)
    recon_loss = F.cross_entropy(y_hat.view(-1, vocab_size), y.view(-1), reduction='sum') / batch_size
    # KLD
    qy_softmax = F.softmax(qy, dim=-1).reshape(*qy.size())
    log_ratio = torch.log(qy_softmax * categorical_dim - eps)  # qy * (log_qy - log (1/N))
    KLD = torch.sum(qy_softmax * log_ratio, dim=-1).mean()
    return recon_loss + KLD, recon_loss, KLD
    # ELBO = -(sum(y * log(y_hat)) - KLD)
    # ELBO = -sum(y * log(y_hat)) + KLD # removing negative
    # ELBO = cross_entropy(y, y_hay) + KLD


def get_all_results(hyperparameters: dict, current_epoch: int, train_loss: float, kld_loss: float, recon_loss: float):
    result = {}
    for name, values in hyperparameters.items():
        result[name] = values
    result['current_epoch'] = current_epoch
    result['train_loss'] = train_loss
    result['kld'] = kld_loss
    result['recon_loss'] = recon_loss
    return result


def train(emb_dim: int, h_dim: int, latent_dim: int, categorical_dim: int, batch_size: int,
          initial_temp: float, min_temp: float, epochs: int, anneal_rate: float):
    hyperparameters = locals()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    preprocessed_data = preprocessing("data/dataset_sentence_level.csv", device)
    vocab = preprocessed_data.vocab
    dataset = preprocessed_data.dataset
    model = BeliefAutoencoder(emb_dim=emb_dim, h_dim=h_dim, vocab=vocab, latent_dim=latent_dim,
                              categorical_dim=categorical_dim, device=device)
    model.to(device)
    train_dataloader = DataLoader(dataset, batch_size=batch_size)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    temp = initial_temp
    results = []
    for epoch in range(epochs):

        train_loss = 0.
        recon_loss = 0.
        kld_loss = 0.
        for batch_idx, batch in tqdm.tqdm(enumerate(train_dataloader), total=len(train_dataloader)):
            x, y, seq_lens = batch
            optimizer.zero_grad()
            y_hat, qy = model(x, seq_lens, temperature=temp)
            loss, recon, kld = loss_function(y=y, y_hat=y_hat, qy=qy, categorical_dim=categorical_dim,
                                             vocab_size=len(vocab))
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            recon_loss += recon.item()
            kld_loss += kld.item()

            if batch_idx % 100 == 1:
                temp = np.maximum(temp * np.exp(-anneal_rate * batch_idx), min_temp)
        print(f"Epoch {epoch} - Train loss {train_loss / len(train_dataloader):.4f} - Temp {temp:.4f}")
        # results.append({'h_dim': h_dim, 'emb_dim':emb_dim, 'epoch':epoch, ''})
        epoch_result = get_all_results(hyperparameters=hyperparameters,
                                       current_epoch=epoch,
                                       train_loss=train_loss / len(train_dataloader),
                                       kld_loss=kld_loss / len(train_dataloader),
                                       recon_loss=recon_loss / len(train_dataloader))
        results.append(epoch_result)

    return pd.DataFrame(results)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--emb_dim", type=int, default=50, help="Dimension of embedding layer")
    parser.add_argument("--batch_size", type=int, default=128, help="Batch Size")
    parser.add_argument("--h_dim", type=int, default=50, help="RNN hidden dim")
    parser.add_argument("--epochs", type=int, default=100, help="number of epochs to train")
    parser.add_argument("--anneal_rate", type=float, default=0.00003, help="Gumbel anneal rate")
    parser.add_argument("--initial_temp", type=int, default=1, help="Gumbel initial temperature")
    parser.add_argument("--min_temp", type=float, default=0.5, help="Gumbel min temperature")
    parser.add_argument("--latent_dim", type=int, default=15, help="Dimension of latent vector")
    parser.add_argument("--categorical_dim", type=int, default=2, help="Number of categories")

    args = parser.parse_args()
    set_seed()
    df_results = train(emb_dim=args.emb_dim,
                       h_dim=args.h_dim,
                       batch_size=args.batch_size,
                       epochs=args.epochs,
                       initial_temp=args.initial_temp,
                       min_temp=args.min_temp,
                       latent_dim=args.latent_dim,
                       categorical_dim=args.categorical_dim,
                       anneal_rate=args.anneal_rate)

    df_results.to_csv('train_results/results.csv', index=False)
