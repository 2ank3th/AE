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
trainset_unlabeled = pickle.load(open("train_unlabeled.p", "rb"))

def bi(layer, size, name,obj):
    #temp = torch.randn(size).type(dtype)
    #tv = Variable(temp)
    #tv.register_parameter(
    para = nn.Parameter(torch.randn(size).type(dtype))
    obj.register_parameter(str(layer)+name+str(size),para)
    return para
    #return Variable(inits * torch.ones(size))


def wi(layer,shape, name,obj):
    para = nn.Parameter(torch.randn(shape[0],shape[1]).type(dtype))
    #print(str(layer)+name + str(shape[0]) + str(shape[1])) #TODO revisit name logic to avoid conflicts
    obj.register_parameter(str(layer)+name + str(shape[0]) + str(shape[1]),para)
    #return Variable(torch.randn(shape[0],shape[1])) / math.sqrt(shape[0])
    return para

shapes = zip(layer_sizes[:-1], layer_sizes[1:])  # shapes of linear layers


join = lambda l, u: torch.cat([l, u], 0)

kwargs = {'num_workers': 1, 'pin_memory': True} if args.cuda else {}

print(trainset_labeled.train_data.size())
train_loader = torch.utils.data.DataLoader(trainset_labeled , batch_size=64, shuffle=True, **kwargs)
unlabeled_train_loader = torch.utils.data.DataLoader( trainset_unlabeled, batch_size=64, shuffle=True, **kwargs)

#valid_loader = torch.utils.data.DataLoader(validset, batch_size=64, shuffle=True)

reconstruction_function = nn.BCELoss()
reconstruction_function.size_average = False

class VAE(nn.Module):
    def __init__(self):
        super(VAE, self).__init__()
              
        self.weights = {'W': [wi(i,s, "W",self) for i,s in enumerate(shapes)],  # Encoder weights
           'V': [wi(i,s[::-1], "V",self) for i,s in enumerate(shapes)],  # Decoder weights
           # batch normalization parameter to shift the normalized value
           'beta': [bi(l, layer_sizes[l+1], "beta",self) for l in range(L)],
           # batch normalization parameter to scale the normalized value
           'gamma': [bi(l, layer_sizes[l+1], "beta",self) for l in range(L)]}

    def split_input(self, h, labeled):
        if labeled:
            return h, h
        else:
            return None, h

    def g_gauss(self,z_c, u, size):
        print("sizze" + str(size))
        "gaussian denoising function proposed in the original paper"
        wi = lambda inits, name: Variable( torch.ones(1,size).expand(u.size()[0],size), requires_grad = True) #TODO possible reason of bad accuracy
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

        print("u" + str(u.size()))
        print("a2" + str(a2.size()))
        temp1 = a2 * u
        temp2 = temp1 + a3
        temp3 = a4 * u

        mu = a1 * nn.functional.sigmoid(temp2) + temp3 + a5
        v = a6 * nn.functional.sigmoid(a7 * u + a8) + a9 * u + a10
        # mu = a1 * tf.sigmoid(a2 * u + a3) + a4 * u + a5
        # v = a6 * tf.sigmoid(a7 * u + a8) + a9 * u + a10

        z_est = (z_c - mu) * v + mu
        return z_est


    def encoder(self,inputs, noise_std, labeled):
        h = inputs + Variable(torch.mul(torch.randn(inputs.size()),noise_std))

        d = {'z': {}, 'm': {}, 'v': {}, 'h': {}}

        d['z'][0] = h

        for l in xrange(1,L+1):

            print("Layer " + str(l)+ ": "+ str(layer_sizes[l - 1])+ " -> "+ str(layer_sizes[l]))

            d['h'][l - 1] = h

            
            z_pre = torch.mm(h, self.weights['W'][l - 1])  # pre-activation

            z_pre_l, z_pre_u = self.split_input(z_pre,labeled)  # split labeled and unlabeled examples

            #TODO why only for unlabbeled if not labeled:
            m, v = torch.mean(z_pre_u, 1), torch.var(z_pre_u, 1)

            print("Mean : "+ str(m))
            print("Var : " + str(v))

            d['m'][l], d['v'][l] = m, v  # save mean and variance of unlabeled examples for decoding

            batch_norm = torch.nn.BatchNorm1d(z_pre.size()[1])
         
            z = batch_norm(z_pre)
            #TODO: BN for test is different

            if l == L:
                # use softmax activation in output layer
                softmax = torch.nn.Softmax()

                h = softmax(z)
                #TODO h = softmax(self.weights['gamma'][l - 1] * (z + self.weights["beta"][l - 1]))
            else:
                # use ReLU activation in hidden layers
                relU = torch.nn.ReLU()

                h = relU(z) #TODO + self.weights["beta"][l - 1])

            d['z'][l] = z


        d['h'][l] = self.split_input(h,labeled)

        return h, d


    def decoder(self,h_clean,h_corr,  d_clean, d_corr):
        # Decoder
        z_est = {}
        d_cost = Variable(torch.FloatTensor(L + 1), requires_grad=False) # to store the denoising cost of all layers

        for l in range(L, 0, -1): #TODO: last layer not run
            print ("Layer ", l, ": ", layer_sizes[l + 1] if l + 1 < len(layer_sizes) else None, " -> ", layer_sizes[
                l], ", denoising cost: ", denoising_cost[l])

            z, z_c = d_clean['z'][l], d_corr['z'][l]

            m, v = d_clean['m'].get(l, 0), d_clean['v'].get(l, 1 - 1e-10)

            if l == L:
                u =  h_corr #unlabeled(h_corr)
            else:
                u = torch.mm(z_est[l + 1], self.weights['V'][l])
                # u = tf.matmul(z_est[l+1], weights['V'][l])

            batch_norm = torch.nn.BatchNorm1d(u.size()[1])

            u = batch_norm(u)
            print(m)

            z_est[l] = self.g_gauss(z_c, u, layer_sizes[l])
            z_est_bn = (z_est[l] - m.expand_as(z_est[l])) / v.expand_as(z_est[l])

            z = z.detach()

            d_cost[l] = torch.nn.functional.binary_cross_entropy(torch.sigmoid(z_est_bn), torch.sigmoid(z))

        return d_cost

    def forward(self, (x,labeled)):

        h_corr,d_corr = self.encoder(x,0.3,labeled) #TODO: add noise
        h_clean,d_clean = self.encoder(x,0,labeled)

        d_cost = self.decoder(h_clean,h_corr, d_clean,d_corr)
        
        # calculate total unsupervised cost by adding the denoising cost of all layers
        #u_cost = torch.zeros(torch.size(d_cost[0]))

        #for ele in d_cost:
        #    u_cost += ele
        # u_cost = tf.add_n(d_cost)
        print('xxxxxxxxx')
        print("d_cost: " + str(d_cost))
        u_cost = torch.sum(d_cost)
        print('xxxxxxxxx')
        print("u_cost: " + str(u_cost))
        print("u_cost: " + str(u_cost.size()))

        #y_N = labeled(y_c)
        #cost = -torch.mean(tf.torch.sum(outputs * torch.log(y_N), 1))  # supervised cost
        # cost = -tf.reduce_mean(tf.reduce_sum(outputs*tf.log(y_N), 1))  # supervised cost
        #loss = cost + u_cost  # total cost


        return u_cost

model  = VAE()

optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum) #TODO: change to something


def train(epoch):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = Variable(data), Variable(target)
        data = data.view(-1,784) #TODO possible reason of bad accuracy
        optimizer.zero_grad()
        loss = model((data, True))
        loss.backward()
        optimizer.step()

        for batch_idx, (data, target) in enumerate(unlabeled_train_loader):
            data, target = Variable(data), Variable(target)
            data = data.view(-1, 784)  # TODO possible reason of bad accuracy
            optimizer.zero_grad()
            loss = model((data, False))
            loss.backward()
            optimizer.step()


        if batch_idx % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                       100. * batch_idx / len(train_loader), loss.data[0]))



for epoch in range(1, args.epochs + 1):
    train(epoch)
    #test(epoch, valid_loader)
