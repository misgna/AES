import torch
from torch.utils.data import Dataset
from torchtext.data import get_tokenizer

tokenizer = get_tokenizer('basic_english')

class ASAPDataset(Dataset):
    def __init__(self, data, vocab):
        self.data = data
        self.vocab = vocab

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        sample = self.data[index]
        essay = sample[2]
        essay = self.vocab(tokenizer(essay))
        score = sample[3]
        return essay, score

def collate_fn(batch):
    essays, scores = zip(*batch)
    max_length = max([len(entry) for entry in essays])
    padded_essays = []
    for tokens in essays:
        padded_essay = tokens + [0] * (max_length - len(tokens))
        padded_essays.append(padded_essay)
    return torch.tensor(padded_essays, dtype=torch.int64), torch.tensor(scores, dtype=torch.int64)