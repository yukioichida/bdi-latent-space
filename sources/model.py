import torch

import torch.nn as nn

import torch.nn.functional as F
import torch.optim as optim
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence


def sample_gumbel(shape, device, eps=1e-20):
    U = torch.rand(shape, device=device)
    return torch.log(-torch.log(U + eps) + eps)


def gumbel_softmax_sample(logits, temperature, device):
    y = logits + sample_gumbel(logits.size(), device)
    return F.softmax(y / temperature, dim=-1)


def gumbel_softmax(logits, temperature, latent_dim, categorical_dim, hard=False, device='cpu'):
    """
    ST-gumple-softmax
    input: [*, n_class]
    return: flatten --> [*, n_class] an one-hot vector
    """
    y = gumbel_softmax_sample(logits, temperature, device)
    if not hard:
        return y.view(-1, latent_dim * categorical_dim)

    shape = y.size()
    _, ind = y.max(dim=-1)
    y_hard = torch.zeros_like(y).view(-1, shape[-1])
    y_hard.scatter_(1, ind.view(-1, 1), 1)
    # y_hard = y_hard.view(*shape)
    # Set gradients w.r.t. y_hard gradients w.r.t. y
    y_hard = (y_hard - y).detach() + y

    return y_hard.view(-1, latent_dim * categorical_dim)


class BeliefAutoencoder(nn.Module):

    def __init__(self, emb_dim, h_dim, vocab_size, pad_idx, latent_dim, categorical_dim, device):
        super(BeliefAutoencoder, self).__init__()
        # encoder
        self.embedding = nn.Embedding(embedding_dim=emb_dim, num_embeddings=vocab_size, padding_idx=pad_idx)
        self.lstm_encoder = nn.GRU(batch_first=True, hidden_size=h_dim, input_size=emb_dim, bidirectional=True)
        # VAE
        self.gumbel_input = nn.Linear(h_dim * 2, latent_dim * categorical_dim)
        # -- decoder --
        # converte o z em um vetor para ser usado como h_t no decoder lstm
        self.z_embedding = nn.Linear(latent_dim * categorical_dim, h_dim * 2)  # z_t -> h_t
        self.lstm_decoder = nn.GRU(batch_first=True, hidden_size=h_dim, input_size=emb_dim, bidirectional=True)
        self.output_layer = nn.Linear(in_features=h_dim * 2, out_features=vocab_size)  #

        self.latent_dim = latent_dim
        self.categorical_dim = categorical_dim
        self.device = device

    def encode(self, x, seq_len):
        x_emb = self.embedding(x)
        x_pack = pack_padded_sequence(x_emb, seq_len.data.tolist(), batch_first=True)
        x, ht = self.lstm_encoder(x_pack)
        encoded_sequence = ht.view(ht.size(1), ht.size(2) * 2)  # bidirectional -> + <-
        return x, encoded_sequence, x_pack

    def decoder(self, input, z, max_seq_len):
        z_emb = self.z_embedding(z).unsqueeze(0)  # z_emb será usado como h inicial do LSTM decoder
        _, batch_size, hidden_len = z_emb.size()
        hidden = z_emb.view(2, batch_size, int(hidden_len/2))  # h_t
        x, _ = self.lstm_decoder(input, hidden)
        # TODO: incluir tamanho max da sequencia original para calcular o loss
        x, _ = pad_packed_sequence(x, batch_first=True, total_length=max_seq_len)
        x = self.output_layer(x)
        return x

    def forward(self, x, seq_len, temperature):
        # ordering by sequence length
        sorted_lengths, sorted_idx = torch.sort(seq_len, descending=True)
        x = x[sorted_idx]
        batch_size, max_seq_len = x.size()  # [batch_size, maximum seq_len from current batch, dim]
        x, h_t, x_pack = self.encode(x, sorted_lengths)

        q_y = self.gumbel_input(h_t)
        q_y = q_y.view(q_y.size(0), self.latent_dim, self.categorical_dim)

        z = gumbel_softmax(q_y,
                           temperature=temperature,
                           latent_dim=self.latent_dim,
                           categorical_dim=self.categorical_dim,
                           device=self.device)

        x = self.decoder(x_pack, z, max_seq_len)
        return x, q_y
