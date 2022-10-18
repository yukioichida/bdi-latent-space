import torch
import numpy as np
import random
import matplotlib.pyplot as plt
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence


from sources.model import BeliefAutoencoder, gumbel_softmax
from sources.preprocessing import preprocessing, preprocess_sentence

def set_seed(seed=20190827):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
set_seed()

preprocessed_data = preprocessing("data/dataset_sentence_level.csv", device='cpu')
vocab = preprocessed_data.vocab
model = BeliefAutoencoder(emb_dim=128, h_dim=512, latent_dim=16, vocab=vocab, categorical_dim=2, activation='gumbel')

#model.load_state_dict(torch.load("models/belief-autoencoder-bc-YP68fCFu.pth", map_location='cpu'))
#model.load_state_dict(torch.load("models/belief-autoencoder-gumbel-128-512-32-FatzuhGa.pth", map_location='cpu'))
model.load_state_dict(torch.load("models/belief-autoencoder-bc-128-512-32-dvmEhYqV.pth", map_location='cpu'))
model.eval()


sentences = ["you have an apple in your inventory", "apple"]
seq_lens = [len(s.split()) for s in sentences]
max_len = max(seq_lens)
vectorized = []
for s in sentences:
    idxs, tgt, seq_len = preprocess_sentence(s, vocab, max_len)
    vectorized.append(idxs)
tensorized = torch.tensor(vectorized)

seq_len = torch.tensor(seq_lens)


with torch.no_grad():
    y_hat, qy, _ = model(x=tensorized, seq_len=seq_len, temperature=0.5)
#qy = gumbel_softmax(qy, temperature=1e-20,latent_dim=30, categorical_dim=2, hard=True)
#qy = F.sigmoid(qy)

latent = torch.round(qy).squeeze(-1)
latent = latent.view(2, 1, 32)
latent_vectors = latent.detach().cpu().numpy()
for vector in latent_vectors:
    plt.figure()
    plt.imshow(vector, cmap='gray')
    plt.show()

inverted_vocab = {idx: word for word, idx in vocab.items()}

y_hat.argmax(dim=-1)
idxs = y_hat.argmax(dim=-1).detach().cpu().numpy()
idx = model.inference(qy, 20)
for idx in idxs:
    print([inverted_vocab[i] for i in idx])
    print(idx)