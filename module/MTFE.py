import time
import math

import torch
from matplotlib import pyplot as plt
from sklearn.metrics import confusion_matrix
from torch import nn, optim
from torch.nn import functional as F


class MTFE(nn.Module):
    def __init__(self, out_c, **kwargs):
        super(MTFE, self).__init__(**kwargs)
        self.branches = nn.ModuleList()
        for i in range(1, out_c + 1):
            branch = self.create_branch(i)
            self.branches.append(branch)

        self.batch_norm = nn.BatchNorm1d(out_c)

    def forward(self, x):

        branch_outputs = []
        for branch in self.branches:
            branch_outputs.append(branch(x))
        x1 = torch.cat(branch_outputs, dim=1)
        x2 = x1
        x3 = F.relu(x2)
        x4 = self.batch_norm(x3)
        return x4

    def create_branch(self, i):
        branch = nn.Sequential(
            nn.Conv1d(1, 1, kernel_size=2 * i - 1, padding=(2 * i - 1 - 1) // 2, stride=1),
        )
        return branch
