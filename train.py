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

device = 'cuda' if torch.cuda.is_available() else 'cpu'

dataset_df = pd.read_csv("data/dataset_sentence_level.csv")
print(len(dataset_df))
dataset_df.head()
all_sentences = dataset_df['sentence'].values.tolist()
all_words = ['<PAD>', '<SOS>', '<EOS>']  # including special tokens

all_splitted_sentences = []
max_len = 0
for sentences in all_sentences:
    words = sentences.split()
    if len(words) > max_len:
        max_len = len(words)
    all_splitted_sentences.append(words)
    [all_words.append(w) for w in words if w not in all_words]

vocab = {w: idx for idx, w in enumerate(all_words)}

pad_idx = vocab['<PAD>']
sos_idx = vocab['<SOS>']
eos_idx = vocab['<EOS>']
f"Num words: {len(words)} - max sentence length {max_len}"

#
all_input_idx = []
all_target_idx = []
all_sequence_len = []
for i, sentence in enumerate(all_splitted_sentences):
    word_idx = [vocab[w] for w in sentence]
    # including <eos> and <sos> tokens
    input_idx = [sos_idx] + word_idx
    target_idx = word_idx + [eos_idx]
    # padding both sequences
    pad_len = (max_len + 1) - len(word_idx)
    pad_input_idx = input_idx + ([pad_idx] * pad_len)
    pad_target_idx = target_idx + ([pad_idx] * pad_len)

    all_input_idx.append(pad_input_idx)
    all_target_idx.append(pad_target_idx)

    all_sequence_len.append(len(input_idx))

tensor_input = torch.tensor(all_input_idx, device=device)
tensor_output = torch.tensor(all_target_idx, device=device)
tensor_len = torch.tensor(all_sequence_len, device=device)
dataset = TensorDataset(tensor_input, tensor_output, tensor_len)

#
def loss_function(y, y_hat, qy, categorical_dim, eps=1e-20):
    # https://discuss.pytorch.org/t/proper-input-to-loss-function-crossentropy-nll/26663/3
    # [batch_size, seq_len, C] -> [batch_size * seq_len, C]
    batch_size = y.size(0)
    recon_loss = F.cross_entropy(y_hat.view(-1, len(vocab)), y.view(-1), reduction='sum') / batch_size
    # KLD
    qy_softmax = F.softmax(qy, dim=-1).reshape(*qy.size())
    log_ratio = torch.log(qy_softmax * categorical_dim - 1e-20)  # qy * (log_qy - log (1/N))
    KLD = torch.sum(qy_softmax * log_ratio, dim=-1).mean()
    return recon_loss + KLD
    # ELBO = -(sum(y * log(y_hat)) - KL)
    # ELBO = -sum(y * log(y_hat)) + KLD # removing negative
    # ELBO = cross_entropy(y, y_hay) + KLD


h_dim = 50
latent_dim = 15
categorical_dim = 2

model = BeliefAutoencoder(emb_dim=100,
                          h_dim=h_dim,
                          device=device,
                          vocab_size=len(vocab),
                          pad_idx=pad_idx,
                          latent_dim=latent_dim,
                          categorical_dim=categorical_dim)
model.to(device)
train_dataloader = DataLoader(dataset, batch_size=64)
optimizer = optim.Adam(model.parameters(), lr=1e-3)

temp = 1.
temp_min = 0.5
ANNEAL_RATE = 0.00003

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
