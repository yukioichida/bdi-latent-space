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

device = 'cuda' if torch.cuda.is_available() else 'cpu'

dataset_df = pd.read_csv("data/dataset_sentence_level.csv")
print(len(dataset_df))
dataset_df.head()
all_sentences = dataset_df['sentence'].values.tolist()
all_words = ['<PAD>', '<SOS>', '<EOS>'] #including special tokens

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
latent_dim = 15 # N states
categorical_dim = 1  # one-of-K vector

def sample_gumbel(shape, eps=1e-20):
    U = torch.rand(shape)
    return -Variable(torch.log(-torch.log(U + eps) + eps))

def gumbel_softmax_sample(logits, temperature):
    y = logits + sample_gumbel(logits.size())
    return F.softmax(y / temperature, dim=-1)

def gumbel_softmax(logits, temperature, hard=False):
    """
    ST-gumple-softmax
    input: [*, n_class]
    return: flatten --> [*, n_class] an one-hot vector
    """
    y = gumbel_softmax_sample(logits, temperature)
    if not hard:
        return y.view(-1, latent_dim * categorical_dim)

    shape = y.size()
    _, ind = y.max(dim=-1)
    y_hard = torch.zeros_like(y).view(-1, shape[-1])
    y_hard.scatter_(1, ind.view(-1, 1), 1)
    y_hard = y_hard.view(*shape)
    # Set gradients w.r.t. y_hard gradients w.r.t. y
    y_hard = (y_hard - y).detach() + y

    return y_hard.view(-1, latent_dim * categorical_dim)

class Autoencoder(nn.Module):

    def __init__(self):
        super(Autoencoder, self).__init__()
        # encoder
        self.embedding = nn.Embedding(embedding_dim=100, num_embeddings=len(vocab), padding_idx=pad_idx)
        self.lstm_encoder = nn.GRU(batch_first=True, hidden_size=100, input_size=100, bidirectional=True)
        # VAE
        self.gumbel_input = nn.Linear(200, latent_dim*categorical_dim)
        # -- decoder --
        # converte o z em um vetor para ser usado como h_t no decoder lstm
        self.z_embedding = nn.Linear(latent_dim * categorical_dim, 100) # z_t -> h_t
        self.lstm_decoder = nn.GRU(batch_first=True, hidden_size=100, input_size=100)
        self.output_layer = nn.Linear(in_features=100, out_features=len(vocab)) #

    def encode(self, x, seq_len):
        x_emb = self.embedding(x)
        x_pack = pack_padded_sequence(x_emb, seq_len.data.tolist(), batch_first=True)
        x, ht = self.lstm_encoder(x_pack)
        encoded_sequence = ht.view(ht.size(1), ht.size(2) * 2) # bidirectional -> + <-
        return x, encoded_sequence, x_pack

    def decoder(self, input, z, max_seq_len):
        z_emb = self.z_embedding(z).unsqueeze(0) # z_emb será usado como h inicial do LSTM decoder
        hidden = z_emb # h_t
        x, _ = self.lstm_decoder(input, hidden)
        # TODO: incluir tamanho max da sequencia original para calcular o loss
        x, _ = pad_packed_sequence(x, batch_first=True, total_length=max_seq_len)
        x = self.output_layer(x)
        return x

    def forward(self, x, seq_len, temperature):
        # ordering by sequence length
        sorted_lengths, sorted_idx = torch.sort(seq_len, descending=True)
        x = x[sorted_idx]

        batch_size, max_seq_len = x.size() # [batch_size, maximum seq_len from current batch, dim]

        x, encoded_sequence, x_pack = self.encode(x, sorted_lengths)
        q_y = self.gumbel_input(encoded_sequence)
        z = gumbel_softmax(q_y, temperature=temperature)
        x = self.decoder(x_pack, z, max_seq_len)
        return x, q_y


#
def loss_function(y, y_hat, qy, eps=1e-20):

    # https://discuss.pytorch.org/t/proper-input-to-loss-function-crossentropy-nll/26663/3
    # [batch_size, seq_len, C] -> [batch_size * seq_len, C]
    batch_size = y.size(0)
    recon_loss = F.cross_entropy(y_hat.view(-1, len(vocab)), y.view(-1), reduction='sum') / batch_size
    # KLD
    qy_softmax = F.softmax(qy, dim=-1).reshape(*qy.size())
    log_ratio = torch.log(qy_softmax * categorical_dim - 1e-20) # qy * (log_qy - log (1/N))
    KLD = torch.sum(qy_softmax * log_ratio, dim=-1).mean()
    return recon_loss + KLD
    # ELBO = -(sum(y * log(y_hat)) - KL)
    # ELBO = -sum(y * log(y_hat)) + KLD # removing negative
    # ELBO = cross_entropy(y, y_hay) + KLD

model = Autoencoder()
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
        loss = loss_function(y=y, y_hat=y_hat, qy=qy)
        loss.backward()
        optimizer.step()
        train_loss += loss.item()
        if batch_idx % 100 == 1:
            temp = np.maximum(temp * np.exp(-ANNEAL_RATE * batch_idx), temp_min)

        #tqdm_batches.set_description(f"Current train loss {loss.item():.4f}")
    print(f"Epoch {epoch} - Train loss {train_loss / len(train_dataloader):.4f} - Temp {temp:.4f}")

    #break

end = time.time()
duration = end - start
print(f"duration = {duration}")