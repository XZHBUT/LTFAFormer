import math

import torch
from matplotlib import pyplot as plt
from sklearn.metrics import confusion_matrix
from torch import nn, optim
from torch.nn import functional as F

from AttentionBlock import Timedomain, Frequencydomain


class LFTFormer_block(nn.Module):
    def __init__(self, d_in, d_out,  heads=1, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.ECABlock = Timedomain(d_in)

        self.LinearAttentionBlock = Frequencydomain(heads, d_in)
        self.batch_norm = nn.BatchNorm1d(d_out)
        self.convs_L = nn.Conv1d(d_in, d_in, kernel_size=5, padding=2, stride=4)
        self.convs_D = nn.Conv1d(d_in, d_out, kernel_size=1, padding=0, stride=1)

    def forward(self, x):
        freq_Weight = self.ECABlock(x)

        Att_x = self.LinearAttentionBlock(x)

        x1 = self.convs_L(Att_x)

        DouBle_Att = x1 * freq_Weight.expand_as(x1)

        x2 = self.convs_D(DouBle_Att)
        x3 = F.gelu(x2)

        LFTFormer_block_out = self.batch_norm(x3)

        return LFTFormer_block_out
