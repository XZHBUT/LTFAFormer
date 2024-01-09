
import math

import torch
from matplotlib import pyplot as plt
from sklearn.metrics import confusion_matrix
from torch import nn, optim
from torch.nn import functional as F




class AddWeight(nn.Module):
    def __init__(self, epsilon=1e-12):
        super(AddWeight, self).__init__()
        self.epsilon = epsilon
        self.w = nn.Parameter(torch.ones(2, dtype=torch.float32), requires_grad=True)
        self.w_relu = nn.ReLU()

    def forward(self, x):
        w = self.w_relu(self.w)
        weight = w / (torch.sum(w, dim=0) + self.epsilon)
        return weight[0] * x[0] + weight[1] * x[1]


class TimeSelfattention(nn.Module):
    def __init__(self, emb_dim, heads):
        super(TimeSelfattention, self).__init__()
        self.dim, self.heads = emb_dim, heads

    def forward(self, q, k, v):
        q = torch.flatten(q, 2).transpose(1, 2)
        k = torch.flatten(k, 2).transpose(1, 2)
        v = torch.flatten(v, 2).transpose(1, 2)
        if self.heads == 1:
            q, k = F.softmax(q, dim=2), F.softmax(k, dim=1)
            return q.bmm(k.transpose(2, 1).bmm(v)).transpose(1, 2)

        else:
            q = q.split(self.dim // self.heads, dim=2)
            k = k.split(self.dim // self.heads, dim=2)
            v = v.split(self.dim // self.heads, dim=2)
            atts = []
            for i in range(self.heads):
                att = F.softmax(q[i], dim=2).bmm(F.softmax(k[i], dim=1).transpose(2, 1).bmm(v[i]))
                atts.append(att.transpose(1, 2))
            return torch.cat(atts, dim=1)


class FFN(nn.Module):
    def __init__(self, dim, ratio=4):
        super(FFN, self).__init__()
        self.MLP = nn.Sequential(
            nn.Linear(dim, dim // ratio), nn.GELU(),
            nn.Linear(dim // ratio, dim), nn.GELU(), )
        self.add = AddWeight()
        self.bn = nn.BatchNorm1d(dim)
    def forward(self, x):
        feature = self.MLP(x.transpose(1, 2))
        return self.bn(self.add([feature.transpose(1, 2), x]))


class Frequencydomain(nn.Module):
    def __init__(self, in_C, gamma=2, b=1, *args, **kwargs):
        super().__init__(*args, **kwargs)
        t = int(abs((math.log(in_C, 2) + b) / gamma))
        k = t if t % 2 else t + 1
        # k = 3
        print(k)
        self.avg_pool1d = nn.AdaptiveAvgPool1d(1)
        self.conv = nn.Conv1d(1, 1, kernel_size=k, padding=int(k / 2), bias=False)

    def forward(self, x):
        y = self.avg_pool1d(x)
        y = self.conv(y.transpose(-1, -2))
        y = y.transpose(-1, -2)
        return y


class Timedomain(nn.Module):
    def __init__(self, heads, dim, *args, **kwargs):
        super(Timedomain, self).__init__()
        self.q_k_v = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(dim, dim, 3,
                          stride=1 if i == 0 else 2,
                          padding=1, groups=dim,
                          bias=False),
                nn.BatchNorm2d(dim),
                nn.Conv2d(dim, dim, 1, 1, 0), nn.GELU())
            for i in range(3)])
        self.MHLSA = TimeSelfattention(dim, heads)
        self.add = AddWeight()
        self.bn = nn.BatchNorm1d(dim)
        self.FFN = FFN(dim)

    def forward(self, x):
        # x [1, 8, 1024]
        b, c, l = x.size()
        maps = x.view(-1, c, int(l ** 0.5), int(l ** 0.5))
        MHLSA = self.MHLSA(
            self.q_k_v[0](maps),
            self.q_k_v[1](maps),
            self.q_k_v[2](maps))
        att_Out = self.bn(self.add([MHLSA, x]))
        FNN_out = self.FFN(att_Out)

        return FNN_out
