import random
import warnings
import os
import time
from tqdm import tqdm, trange


import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch.utils.data import DataLoader

import matplotlib.pyplot as plt

from utils import MyDataset, trainTestSplit, preprocess, ProcessInputData

# from evaluation import CCCscore, output_eval

use_cuda = torch.cuda.is_available()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
warnings.filterwarnings("ignore")

##################################################### Basic network modes ######################################################################
##################################################### Basic network modes ######################################################################
##################################################### Basic network modes ######################################################################
# Define models

class AutoEncoder(nn.Module):
    def __init__(self, idm, shape, k):
        super(AutoEncoder, self).__init__()
        self.name = 'ae'
        self.state = 'train'  # or 'test'
        self.inputdim = idm
        self.shape = shape

        self.encoder = nn.Sequential(
            nn.Dropout(p=0.3),
            nn.Linear(self.inputdim, 512),
            nn.CELU(),
            nn.BatchNorm1d(512),

            nn.Dropout(p=0.3),
            nn.Linear(512, 256),
            nn.CELU(),
            nn.BatchNorm1d(256),

            nn.Dropout(p=0.3),
            nn.Linear(256, 128),
            nn.CELU(),
            nn.BatchNorm1d(128)
        )

        self.predicter = nn.Sequential(
            # nn.Dropout(p=0.3),
            nn.Linear(self.shape, 64),
            nn.CELU(),
            nn.BatchNorm1d(64),

            nn.Linear(64, k)

        )

        self.decoder = nn.Sequential(nn.Linear(k, 64, bias=False),
                                     nn.Linear(64, 128, bias=False),
                                     nn.Linear(128, 256, bias=False),
                                     nn.Linear(256, 512, bias=False),
                                     nn.Linear(512, self.inputdim, bias=False))

    def encode(self, x):
        return self.encoder(x)

    def predict(self, y):
        return self.predicter(y)

    def decode(self, z):
        return self.decoder(z)

    def refraction(self, x):
        x_sum = torch.sum(x, dim=1, keepdim=True)
        return x / x_sum

    def sigmatrix(self):
        w0 = (self.decoder[0].weight.T)
        w1 = (self.decoder[1].weight.T)
        w2 = (self.decoder[2].weight.T)
        w3 = (self.decoder[3].weight.T)
        w4 = (self.decoder[4].weight.T)
        w01 = (torch.mm(w0, w1))
        w02 = (torch.mm(w01, w2))
        w03 = (torch.mm(w02, w3))
        w04 = (torch.mm(w03, w4))
        return F.hardtanh(w04, 0, 1)

    def forward(self, x):
        sigmatrix = self.sigmatrix()
        z_dis = self.encode(x)
        z = self.predict(z_dis)

        if self.state == 'train':
            z = F.hardtanh(z, 0, 1)
        elif self.state == 'test':
            z = F.hardtanh(z, 0, 1)
            z = self.refraction(z)

        x_recon = torch.mm(z, sigmatrix)
        return x_recon, z_dis, z, sigmatrix


class discriminator(nn.Module):
    def __init__(self, shape):
        super(discriminator, self).__init__()
        self.shape = shape
        self.restored = False

        self.discrim = nn.Sequential(
            nn.Linear(self.shape, 32),
            nn.LeakyReLU(),

            nn.Linear(32, 2),
            nn.LogSoftmax()
        )

    def forward(self, x):
        out = self.discrim(x)
        return out
