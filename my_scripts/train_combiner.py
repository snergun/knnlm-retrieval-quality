import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset

class Combiner(nn.Module):
    def __init__(self, input_size, output_size):
        super(Combiner, self).__init__()
        self.fc1 = nn.Linear(input_size, output_size)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.5)
        self.fc2 = nn.Linear(output_size, output_size)

    def forward(self, x):
        return self.fc(x)