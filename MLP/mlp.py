import torch
from torch import nn

class MLP(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, num_classes):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.linear1 = nn.Linear(embedding_dim, hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, num_classes)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, essay):
        embedded = self.embedding(essay)
        hidden = torch.mean(embedded, dim=1)
        layer1 = self.linear1(hidden)
        layer1_relu = self.relu(layer1)
        layer2 = self.linear2(layer1_relu)
        output = self.sigmoid(layer2)
        return output