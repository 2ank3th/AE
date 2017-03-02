from __future__ import print_function

import torch
from torch.autograd import Variable
import input_data
import argparse
import math
import os
import csv
import pickle

import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms



from tqdm import tqdm

dtype = torch.FloatTensor
layer_sizes = [784, 1000, 500, 250, 250, 250, 10]
L = len(layer_sizes) - 1

# Training settings
parser = argparse.ArgumentParser(description='PyTorch MNIST Example')
parser.add_argument('--batch-size', type=int, default=64, metavar='N',
                    help='input batch size for training (default: 64)')
parser.add_argument('--test-batch-size', type=int, default=1000, metavar='N',
                    help='input batch size for testing (default: 1000)')
parser.add_argument('--epochs', type=int, default=1, metavar='N',
                    help='number of epochs to train (default: 10)')
parser.add_argument('--lr', type=float, default=0.01, metavar='LR',
                    help='learning rate (default: 0.01)')
parser.add_argument('--momentum', type=float, default=0.5, metavar='M',
                    help='SGD momentum (default: 0.5)')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='enables CUDA training')
parser.add_argument('--seed', type=int, default=1, metavar='S',
                    help='random seed (default: 1)')
parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                    help='how many batches to wait before logging training status')
args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()


# hyperparameters that denote the importance of each layer
denoising_cost = [1000.0, 10.0, 0.10, 0.10, 0.10, 0.10, 0.10]


torch.manual_seed(args.seed)
if args.cuda:
    torch.cuda.manual_seed(args.seed)


print('loading data!')
trainset_labeled = pickle.load(open("train_labeled.p", "rb"))
#validset = pickle.load(open("validation.p", "rb"))
#trainset_unlabeled = pickle.load(open("train_unlabeled_small.p", "rb"))

def bi(inits, size, name,obj):
    #temp = torch.randn(size).type(dtype)
    #tv = Variable(temp)
    #tv.register_parameter(
    para = nn.Parameter(torch.randn(size).type(dtype))
    obj.register_parameter(name+str(size),para)
    return para
    #return Variable(inits * torch.ones(size))


def wi(shape, name,obj):
    para = nn.Parameter(torch.randn(shape[0],shape[1]).type(dtype))
    print(name + str(shape[0]) + str(shape[1])) #TODO revisit name logic to avoid conflicts
    obj.register_parameter(name + str(shape[0]) + str(shape[1]),para)
    return para
    #return Variable(torch.randn(shape[0],shape[1])) / math.sqrt(shape[0])

shapes = zip(layer_sizes[:-1], layer_sizes[1:])  # shapes of linear layers


join = lambda l, u: torch.cat([l, u], 0)
labeled = lambda x: torch.narrow(x, [0, 0], [batch_size, -1]) if x is not None else x
unlabeled = lambda x: torch.narrow(x, [batch_size, 0], [-1, -1]) if x is not None else x
split_lu = lambda x: (x, x ) #TODO split input



kwargs = {'num_workers': 1, 'pin_memory': True} if args.cuda else {}

print(trainset_labeled.train_data.size())
train_loader = torch.utils.data.DataLoader(trainset_labeled, batch_size=101, shuffle=True, **kwargs)
#valid_loader = torch.utils.data.DataLoader(validset, batch_size=64, shuffle=True)


class VAE(nn.Module):
    def __init__(self):
        super(VAE, self).__init__()
              
        self.weights = {'W': [wi(s, "W",self) for s in shapes],  # Encoder weights
           'V': [wi(s[::-1], "V",self) for s in shapes],  # Decoder weights
           # batch normalization parameter to shift the normalized value
           'beta': [bi(0.0, layer_sizes[l+1], "beta",self) for l in range(L)],
           # batch normalization parameter to scale the normalized value
           'gamma': [bi(1.0, layer_sizes[l+1], "beta",self) for l in range(L)]}

    def g_gauss(z_c, u, size):
        "gaussian denoising function proposed in the original paper"
        wi = lambda inits, name: torch.Variable(inits * torch.ones([size]), name=name)
        # wi = lambda inits, name: tf.Variable(inits * tf.ones([size]), name=name)
        a1 = wi(0., 'a1')
        a2 = wi(1., 'a2')
        a3 = wi(0., 'a3')
        a4 = wi(0., 'a4')
        a5 = wi(0., 'a5')

        a6 = wi(0., 'a6')
        a7 = wi(1., 'a7')
        a8 = wi(0., 'a8')
        a9 = wi(0., 'a9')
        a10 = wi(0., 'a10')

        mu = a1 * torch.nn.sigmoid(a2 * u + a3) + a4 * u + a5
        v = a6 * torch.nn.sigmoid(a7 * u + a8) + a9 * u + a10
        # mu = a1 * tf.sigmoid(a2 * u + a3) + a4 * u + a5
        # v = a6 * tf.sigmoid(a7 * u + a8) + a9 * u + a10

        z_est = (z_c - mu) * v + mu
        return z_est


    def encoder(self,inputs, noise_std):
        print("input size: " + str(inputs.size()))
        print("input data size: " + str(inputs.data.size()))
        h = Variable(inputs.data + torch.mul(torch.randn(inputs.size()),noise_std))
        d = {}
        d['labeled'] = {'z': {}, 'm': {}, 'v': {}, 'h': {}}
        d['unlabeled'] = {'z': {}, 'm': {}, 'v': {}, 'h': {}}

        d['labeled']['z'][0], d['unlabeled']['z'][0] = split_lu(h)

        for l in xrange(1,L+1):
            print("Layer " + str(l)+ ": "+ str(layer_sizes[l - 1])+ " -> "+ str(layer_sizes[l]))
            d['labeled']['h'][l - 1], d['unlabeled']['h'][l - 1] = split_lu(h)
            print("weight: " + str(self.weights['W'][l - 1].size()))
            print("h: " + str(h.size()))
            
            z_pre = torch.mm(h, self.weights['W'][l - 1])  # pre-activation
            z_pre_l, z_pre_u = split_lu(z_pre)  # split labeled and unlabeled examples
            m, v = torch.mean(z_pre_u.data, 1), torch.var(z_pre_u.data, 1)

            print("tensor size :" + str(z_pre.size()))
            batch_norm = torch.nn.BatchNorm1d(z_pre.size()[1])
         
            # perform batch normalization according to value of boolean "training" placeholder:
            z = batch_norm(z_pre)

            if l == L:
                # use softmax activation in output layer
                softmax = torch.nn.Softmax()

                h = softmax(self.weights['gamma'][l - 1] * (z + self.weights["beta"][l - 1]))
            else:
                # use ReLU activation in hidden layers
                relU = torch.nn.ReLU()
                
                print("z: " + str(z.size()))
                print("beta weight: " + str(self.weights["beta"][l - 1].size()))
        
                h = relU(z + self.weights["beta"][l - 1])
                print("h size in hidden layer: " + str(h.size()))
            d['labeled']['z'][l], d['unlabeled']['z'][l] = split_lu(z)
            d['unlabeled']['m'][l], d['unlabeled']['v'][l] = m, v  # save mean and variance of unlabeled examples for decoding
        d['labeled']['h'][l], d['unlabeled']['h'][l] = split_lu(h)
        return h, d


    def decoder(self,y_clean,y_corr,  clean, corr):
        # Decoder
        z_est = {}
        d_cost = []  # to store the denoising cost of all layers
        for l in range(L, -1, -1):
            print ("Layer ", l, ": ", layer_sizes[l + 1] if l + 1 < len(layer_sizes) else None, " -> ", layer_sizes[
                l], ", denoising cost: ", denoising_cost[l])
            z, z_c = clean['unlabeled']['z'][l], corr['unlabeled']['z'][l]
            m, v = clean['unlabeled']['m'].get(l, 0), clean['unlabeled']['v'].get(l, 1 - 1e-10)
            if l == L:
                u = unlabeled(y_corr)
            else:
                u = torch.matmul(z_est[l + 1], self.weights['V'][l])
                # u = tf.matmul(z_est[l+1], weights['V'][l])

            batch_norm = torch.nn.BatchNorm2d(64*u.size[0]*u.size[1])

            u = batch_norm(u)

            z_est[l] = self.g_gauss(z_c, u, layer_sizes[l])
            z_est_bn = (z_est[l] - m) / v
            # append the cost of this layer to d_cost
            d_cost.append(
                (torch.mean(torch.sum((z_est_bn - z) * (z_est_bn - z), 1)) / layer_sizes[l]) * denoising_cost[l])
            # d_cost.append((tf.reduce_mean(tf.reduce_sum(tf.square(z_est_bn - z), 1)) / layer_sizes[l]) * denoising_cost[l])
        return d_cost

    def forward(self, x):
        h1,d1 = self.encoder(x,0) #TODO: add noise
        h2, d2 = self.encoder(x,0)

        d_cost = self.decoder(h1, d1)

        # calculate total unsupervised cost by adding the denoising cost of all layers
        u_cost = torch.zeros(torch.size(d_cost[0]))
        for ele in d_cost:
            u_cost += ele
        # u_cost = tf.add_n(d_cost)

        y_N = labeled(y_c)
        cost = -torch.mean(tf.torch.sum(outputs * torch.log(y_N), 1))  # supervised cost
        # cost = -tf.reduce_mean(tf.reduce_sum(outputs*tf.log(y_N), 1))  # supervised cost
        loss = cost + u_cost  # total cost


        return

model  = VAE()

optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum) #TODO: change to something

def train(epoch):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = Variable(data), Variable(target)
        data = data.view(-1,784) #TODO possible reason of bad accuracy
        optimizer.zero_grad()
        output = model(data)
        loss = F.nll_loss(output, target)
        loss.backward()
        optimizer.step()
        if batch_idx % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                       100. * batch_idx / len(train_loader), loss.data[0]))



for epoch in range(1, args.epochs + 1):
    train(epoch)
    #test(epoch, valid_loader)
