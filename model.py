

import torch

from torch import nn, optim
from torch.nn import functional as F
from module.MTFE import MTFE




from module.LTFAFormerBlock import LFTFormer_block

from module.Classifier import MLPClassifier


class LFTFormer(nn.Module):
    def __init__(self, *args, **kwargs):
        super(LFTFormer, self).__init__(*args, **kwargs)
        self.MTFE1 = MTFE(8)  # torch.Size([1, 8, 1024])
        self.Encoder = nn.Sequential(
            LFTFormer_block(8, 16, heads=1),  # torch.Size([1, 16, 256])
            LFTFormer_block(16, 32, heads=1),  # torch.Size([1, 32, 64])
            LFTFormer_block(32, 64, heads=1),  # torch.Size([1, 64, 16])
            nn.AdaptiveAvgPool1d(1)  # torch.Size([1, 64, 1])
        )
        self.Feedforward = nn.Sequential(
            nn.Linear(64, 64),
            nn.BatchNorm1d(64),
            nn.GELU(),
            nn.Dropout(0.3),
        )
        self.lastLinear = nn.Linear(64, 10)

        nn.init.xavier_uniform_(self.Feedforward[0].weight)

        self.zero_last_layer_weight()

    def zero_last_layer_weight(self):
        self.lastLinear.weight.data = torch.zeros_like(self.lastLinear.weight)
        self.lastLinear.bias.data = torch.zeros_like(self.lastLinear.bias)

    def forward(self, data):
        FeatureHead = self.MTFE1(data)
        EncodeFeature = self.Encoder(FeatureHead)  # torch.Size([1, 64, 1])
        EncodeFeature = EncodeFeature.squeeze(2)  # torch.Size([1, 64])
        x1 = self.Feedforward(EncodeFeature)
        out = self.lastLinear(x1)
        return out




