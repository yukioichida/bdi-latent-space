import time
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from torch.autograd import Variable
import torch.nn.functional as F
import torch.optim as optim
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

import tqdm

from sources.model import BeliefAutoencoder
from sources.preprocessing import preprocessing

device = 'cuda' if torch.cuda.is_available() else 'cpu'

preprocessed_data = preprocessing("data/dataset_sentence_level.csv", device)
dataset = preprocessed_data.dataset
vocab_size = len(preprocessed_data.vocab)
pad_idx = preprocessed_data.pad_idx

#
def loss_function(y, y_hat, qy, categorical_dim, eps=1e-20):
    batch_size = y.size(0)
    recon_loss = F.cross_entropy(y_hat.view(-1, vocab_size), y.view(-1), reduction='mean') #/ batch_size
    # KLD
    qy_softmax = F.softmax(qy, dim=-1).reshape(*qy.size())
    log_ratio = torch.log(qy_softmax * categorical_dim - 1e-20)  # qy * (log_qy - log (1/N))
    KLD = torch.sum(qy_softmax * log_ratio, dim=-1).mean()
    return recon_loss + KLD
    # ELBO = -(sum(y * log(y_hat)) - KL)
    # ELBO = -sum(y * log(y_hat)) + KLD # removing negative
    # ELBO = cross_entropy(y, y_hay) + KLD


emb_dim = 100
h_dim = 50
latent_dim = 15
categorical_dim = 2

model = BeliefAutoencoder(emb_dim=emb_dim,
                          h_dim=h_dim,
                          device=device,
                          vocab_size=vocab_size,
                          pad_idx=pad_idx,
                          latent_dim=latent_dim,
                          categorical_dim=categorical_dim)
model.to(device)
train_dataloader = DataLoader(dataset, batch_size=64)
optimizer = optim.Adam(model.parameters(), lr=1e-3)

temp = 1.
temp_min = 0.5
#ANNEAL_RATE = 0.00003
ANNEAL_RATE = 0.0003

start = time.time()
EPOCH = 20
for epoch in range(EPOCH):
    tqdm_batches = tqdm.tqdm(enumerate(train_dataloader), total=len(train_dataloader))

    train_loss = 0
    for batch_idx, batch in tqdm_batches:
        x, y, seq_lens = batch
        optimizer.zero_grad()
        y_hat, qy = model(x, seq_lens, temperature=temp)
        loss = loss_function(y=y, y_hat=y_hat, qy=qy, categorical_dim=categorical_dim)
        loss.backward()
        optimizer.step()
        train_loss += loss.item()
        if batch_idx % 100 == 1:
            temp = np.maximum(temp * np.exp(-ANNEAL_RATE * batch_idx), temp_min)

        # tqdm_batches.set_description(f"Current train loss {loss.item():.4f}")
    print(f"Epoch {epoch} - Train loss {train_loss / len(train_dataloader):.4f} - Temp {temp:.4f}")

    # break

end = time.time()
duration = end - start
print(f"duration = {duration}")
