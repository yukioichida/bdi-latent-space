import time
import argparse
import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
import tqdm
from torch.utils.data import DataLoader

from sources.model import BeliefAutoencoder
from sources.preprocessing import preprocessing


#
def loss_function(y, y_hat, qy, categorical_dim, vocab_size, eps=1e-20):
    batch_size = y.size(0)
    recon_loss = F.cross_entropy(y_hat.view(-1, vocab_size), y.view(-1), reduction='sum') / batch_size
    # KLD
    qy_softmax = F.softmax(qy, dim=-1).reshape(*qy.size())
    log_ratio = torch.log(qy_softmax * categorical_dim - 1e-20)  # qy * (log_qy - log (1/N))
    KLD = torch.sum(qy_softmax * log_ratio, dim=-1).mean()
    return recon_loss + KLD
    # ELBO = -(sum(y * log(y_hat)) - KL)
    # ELBO = -sum(y * log(y_hat)) + KLD # removing negative
    # ELBO = cross_entropy(y, y_hay) + KLD


def train(args):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    preprocessed_data = preprocessing("data/dataset_sentence_level.csv", device)
    dataset = preprocessed_data.dataset
    vocab_size = len(preprocessed_data.vocab)
    pad_idx = preprocessed_data.pad_idx

    emb_dim = args.emb_dim
    h_dim = args.h_dim
    latent_dim = args.latent_dim
    categorical_dim = args.categorical_dim

    model = BeliefAutoencoder(emb_dim=emb_dim,
                              h_dim=h_dim,
                              device=device,
                              vocab_size=vocab_size,
                              pad_idx=pad_idx,
                              latent_dim=latent_dim,
                              categorical_dim=categorical_dim)
    model.to(device)
    train_dataloader = DataLoader(dataset, batch_size=args.batch_size)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    temp = args.initial_temp
    temp_min = args.min_temp
    # ANNEAL_RATE = 0.00003
    ANNEAL_RATE = args.anneal_rate

    start = time.time()
    for epoch in range(args.epochs):

        train_loss = 0.
        for batch_idx, batch in tqdm.tqdm(enumerate(train_dataloader), total=len(train_dataloader)):
            x, y, seq_lens = batch
            optimizer.zero_grad()
            y_hat, qy = model(x, seq_lens, temperature=temp)
            loss = loss_function(y=y, y_hat=y_hat, qy=qy, categorical_dim=categorical_dim, vocab_size=vocab_size)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            if batch_idx % 100 == 1:
                temp = np.maximum(temp * np.exp(-ANNEAL_RATE * batch_idx), temp_min)
        print(f"Epoch {epoch} - Train loss {train_loss / len(train_dataloader):.4f} - Temp {temp:.4f}")

        # break

    end = time.time()
    duration = end - start
    print(f"duration = {duration}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--emb_dim", type=int, default=50, help="Dimension of embedding layer")
    parser.add_argument("--batch_size", type=int, default=64, help="Batch Size")
    parser.add_argument("--h_dim", type=int, default=50, help="RNN hidden dim")
    parser.add_argument("--epochs", type=int, default=4, help="number of epochs to train")
    parser.add_argument("--anneal_rate", type=float, default=0.00003, help="Gumbel anneal rate")
    parser.add_argument("--initial_temp", type=int, default=1, help="Gumbel initial temperature")
    parser.add_argument("--min_temp", type=float, default=0.5, help="Gumbel min temperature")
    parser.add_argument("--latent_dim", type=int, default=15, help="Dimension of latent vector")
    parser.add_argument("--categorical_dim", type=int, default=2, help="Number of categories")

    args = parser.parse_args()
    train(args)