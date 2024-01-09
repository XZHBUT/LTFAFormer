import math

import torch
from matplotlib import pyplot as plt
from sklearn.metrics import confusion_matrix
from torch import nn, optim
from torch.nn import functional as F


class MLPClassifier(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, dropout_prob, lastWeightZero=True):
        super(MLPClassifier, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, output_dim)
        self.dropout = nn.Dropout(dropout_prob)
        self.BN = nn.BatchNorm1d(hidden_dim)
        if lastWeightZero:
            self.zero_last_layer_weight()

    def zero_last_layer_weight(self):
        self.fc2.weight.data = torch.zeros_like(self.fc2.weight)

    def forward(self, x):
        x = x.squeeze(2)
        x = self.fc1(x)
        x = self.BN(x)
        x = F.gelu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        return x
