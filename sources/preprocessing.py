import pandas as pd
import torch
from torch.utils.data import DataLoader, TensorDataset

from collections import namedtuple
PreprocessedData = namedtuple('PreprocessedData', ['dataset', 'sos_idx', 'pad_idx', 'eos_idx', 'vocab'])

def preprocess_sentence(sentence: str, vocab: dict, max_len: int):
    word_idx = [vocab[w] for w in sentence.split()]
    pad_idx = vocab['<PAD>']
    sos_idx = vocab['<SOS>']
    eos_idx = vocab['<EOS>']
    # including <eos> and <sos> tokens
    input_idx = [sos_idx] + word_idx
    target_idx = word_idx + [eos_idx]
    # padding both sequences
    pad_len = (max_len + 1) - len(word_idx)
    pad_input_idx = input_idx + ([pad_idx] * pad_len)
    pad_target_idx = target_idx + ([pad_idx] * pad_len)
    return pad_input_idx, pad_target_idx, len(pad_input_idx)


def preprocessing(filename: str, device: str) -> PreprocessedData:
    dataset_df = pd.read_csv(filename)
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

    all_input_idx = []
    all_target_idx = []
    all_sequence_len = []
    for i, sentence in enumerate(all_sentences):
        pad_input_idx, pad_target_idx, seq_len = preprocess_sentence(sentence, vocab, max_len)
        all_input_idx.append(pad_input_idx)
        all_target_idx.append(pad_target_idx)
        all_sequence_len.append(seq_len)

    tensor_input = torch.tensor(all_input_idx, device=device)
    tensor_output = torch.tensor(all_target_idx, device=device)
    tensor_len = torch.tensor(all_sequence_len, device=device)
    dataset = TensorDataset(tensor_input, tensor_output, tensor_len)
    return PreprocessedData(dataset=dataset, pad_idx=pad_idx, sos_idx=sos_idx, eos_idx=eos_idx, vocab=vocab)


