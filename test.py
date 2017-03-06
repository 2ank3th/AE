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


a = torch.randn(2, 2)
print(a)

print(torch.norm(a, 2, 0))


print(torch.norm(a, 1, 1))


