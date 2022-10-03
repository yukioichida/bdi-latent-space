import torch

import torch.nn as nn

import torch.nn.functional as F
import torch.optim as optim
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence


def binary_concrete_sample(qy, temperature, device, eps=1e-20):
    U = torch.rand(qy.size(), device=device)
    logistic = torch.log(U + eps) - torch.log(1 - U + eps)
    logits = (qy + logistic) / temperature
    binary_concrete = torch.sigmoid(logits)
    return binary_concrete


def gumbel_softmax_sample(qy, temperature, device, eps=1e-20):
    U = torch.rand(qy.size(), device=device)
    sample_gumbel = torch.log(-torch.log(U + eps) + eps)
    y = (qy + sample_gumbel) / temperature
    gumbel = F.softmax(y, dim=-1)
    return gumbel


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

    def __init__(self, emb_dim, h_dim, vocab, latent_dim, categorical_dim=1, device='cpu', pad_token='<PAD>',
                 activation='gumbel'):
        super(BeliefAutoencoder, self).__init__()
        self.vocab_size = len(vocab)
        # encoder
        self.embedding = nn.Embedding(embedding_dim=emb_dim, num_embeddings=self.vocab_size,
                                      padding_idx=vocab[pad_token])
        self.lstm_encoder = nn.GRU(batch_first=True, hidden_size=h_dim, input_size=emb_dim, bidirectional=True)

        # VAE
        self.sampling_input = nn.Linear(h_dim * 2, latent_dim * categorical_dim)

        # -- decoder --
        # converte o z em um vetor para ser usado como h_t no decoder lstm
        self.z_embedding = nn.Linear(latent_dim * categorical_dim, h_dim * 2)  # z_t -> h_t
        self.lstm_decoder = nn.GRU(batch_first=True, hidden_size=h_dim, input_size=emb_dim, bidirectional=True)
        self.output_layer = nn.Linear(in_features=h_dim * 2, out_features=self.vocab_size)  #

        self.latent_dim = latent_dim
        self.categorical_dim = categorical_dim
        self.device = device
        self.activation = activation

    def encode(self, x, seq_len):
        x_emb = self.embedding(x)
        x_pack = pack_padded_sequence(x_emb, seq_len.data.tolist(), batch_first=True)
        x, ht = self.lstm_encoder(x_pack)
        encoded_sequence = ht.view(ht.size(1), ht.size(2) * 2)  # bidirectional -> + <-
        return x, encoded_sequence, x_pack

    def decoder(self, input, z, max_seq_len):
        z_emb = self.z_embedding(z).unsqueeze(0)  # z_emb será usado como h inicial do LSTM decoder
        _, batch_size, hidden_len = z_emb.size()
        hidden = z_emb.view(2, batch_size, int(hidden_len / 2))  # h_t
        x, _ = self.lstm_decoder(input, hidden)
        x, _ = pad_packed_sequence(x, batch_first=True, total_length=max_seq_len)
        x = self.output_layer(x)
        return x

    def sampling(self, qy, temperature):
        if self.activation == 'gumbel':
            qy = qy.view(qy.size(0), self.latent_dim, self.categorical_dim)
            return gumbel_softmax(qy, temperature, self.latent_dim, self.categorical_dim, device=self.device)
        elif self.activation == 'bc':
            return binary_concrete_sample(qy, temperature, self.device)
        else:
            raise Exception("Invalid sampling activation. Values available: gumbel, bc")

    def forward(self, x, seq_len, temperature):
        # ordering by sequence length
        sorted_lengths, sorted_idx = torch.sort(seq_len, descending=True)
        x = x[sorted_idx]
        batch_size, max_seq_len = x.size()  # [batch_size, maximum seq_len from current batch, dim]

        # encoder
        x, h_t, word_emb = self.encode(x, sorted_lengths)

        # sampling
        qy = self.sampling_input(h_t)
        z = self.sampling(qy, temperature)
        # decoder
        x = self.decoder(word_emb, z, max_seq_len)
        return x, qy

    def loss_function(self, y, y_hat, qy):
        if self.activation == 'gumbel':
            return self._gumbel_loss_function(y, y_hat, qy)
        elif self.activation == 'bc':
            return self._binary_concrete_loss_function(y, y_hat, qy)
        else:
            raise Exception("Invalid sampling activation. Values available: gumbel, bc")

    def _binary_concrete_loss_function(self, y, y_hat, qy, p=0.5, eps=1e-20):
        # KLD
        qy_sigmoid = torch.sigmoid(qy)  # [0~1]
        log_q0 = torch.log(qy_sigmoid + eps)
        log_q1 = torch.log((1 - qy_sigmoid) + eps)
        p = torch.tensor([p], device=self.device)
        log_p0 = torch.log(p + eps)
        log_p1 = torch.log((1 - p) + eps)
        # log ratio of q(x) and p(x)
        loss = qy_sigmoid * (log_q0 - log_p0) + (1 - qy_sigmoid) * (log_q1 - log_p1)
        KLD = torch.sum(loss)
        # recon loss
        recon_loss = F.cross_entropy(y_hat.view(-1, self.vocab_size), y.view(-1), reduction='sum') / y.size(0)

        # ELBO
        loss = KLD + recon_loss
        return loss, recon_loss, KLD

    def _gumbel_loss_function(self, y, y_hat, qy, eps=1e-20):
        batch_size = y.size(0)
        recon_loss = F.cross_entropy(y_hat.view(-1, self.vocab_size), y.view(-1), reduction='sum') / batch_size
        # KLD
        qy_softmax = F.softmax(qy, dim=-1).reshape(*qy.size())
        log_ratio = torch.log(qy_softmax * self.categorical_dim + eps)  # plus epsilon for avoiding log(0)
        KLD = torch.sum(qy_softmax * log_ratio, dim=-1).mean()
        loss = recon_loss + KLD
        return loss, recon_loss, KLD

        # KLD = qy * (log (qy + eps) - log (1/N))
        # KLD = qy * log ((qy + eps) / (1/N))
        # KLD = qy * log (qy * N + eps)

        # ELBO = -(sum(y * log(y_hat)) - KLD)
        # ELBO = -sum(y * log(y_hat)) + KLD # removing negative
        # ELBO = cross_entropy(y, y_hay) + KLD
