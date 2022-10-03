import tqdm

import torch
from torch.utils.data import DataLoader
import numpy as np
import random
import matplotlib.pyplot as plt
import torch.nn.functional as F

from sources.model import BeliefAutoencoder, gumbel_softmax
from sources.preprocessing import preprocessing, preprocess_sentence


def set_seed(seed=20190827):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


set_seed()

preprocessed_data = preprocessing("data/dataset_sentence_level.csv", device='cpu')
vocab = preprocessed_data.vocab
dataset = preprocessed_data.dataset

with torch.no_grad():
    model = BeliefAutoencoder(emb_dim=300, h_dim=300, latent_dim=30, vocab=vocab, categorical_dim=2)
    state_dict = torch.load("models/belief-autoencoder-gumbel-d6HnxdoH.pth", map_location=torch.device('cpu'))
    print(state_dict)
    #model.load_state_dict(state_dict)
    model.eval()

    train_dataloader = DataLoader(dataset, batch_size=128)
    train_loss = 0.
    recon_loss = 0.
    kld_loss = 0.
    for batch_idx, batch in tqdm.tqdm(enumerate(train_dataloader), total=len(train_dataloader)):
        x, y, seq_lens = batch
        y_hat, qy = model(x, seq_lens, temperature=0.5)
        loss, recon, kld = model.loss_function(y=y, y_hat=y_hat, qy=qy)
        train_loss += loss.item()
        recon_loss += recon.item()
        kld_loss += kld.item()

    total_train_loss = train_loss / len(train_dataloader)
    print(total_train_loss)
