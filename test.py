from __future__ import print_function

import torch
from torch.autograd import Variable
import input_data
import argparse
import math
import os
import csv
import cPickle as pickle

import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms


x = torch.rand(2, 2)

print(x)

row = x.size()[0]
column = x.size()[1]

m = torch.mean(x, 0)

print(m.expand(row, column))

