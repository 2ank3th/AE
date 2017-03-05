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



from tqdm import tqdm

dtype = torch.FloatTensor
layer_sizes = [784, 1000, 500, 250, 250, 250, 10]
L = len(layer_sizes) - 1

# Training settings
parser = argparse.ArgumentParser(description='PyTorch MNIST Example')
parser.add_argument('--batch-size', type=int, default=64, metavar='N',
                    help='input batch size for training (default: 64)')
parser.add_argument('--test-batch-size', type=int, default=64, metavar='N',
                    help='input batch size for testing (default: 1000)')
parser.add_argument('--epochs', type=int, default=100, metavar='N',
                    help='number of epochs to train (default: 10)')
parser.add_argument('--lr', type=float, default=0.01, metavar='LR',
                    help='learning rate (default: 0.01)')
parser.add_argument('--momentum', type=float, default=0.5, metavar='M',
                    help='SGD momentum (default: 0.5)')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='enables CUDA training')
parser.add_argument('--seed', type=int, default=1, metavar='S',
                    help='random seed (default: 1)')
parser.add_argument('--log-interval', type=int, default=20, metavar='N',
                    help='how many batches to wait before logging training status')
args = parser.parse_args()


# hyperparameters that denote the importance of each layer
denoising_cost = [1000.0, 10.0, 0.10, 0.10, 0.10, 0.10, 0.10]


torch.manual_seed(args.seed)


print('loading data!')
train_loader = torch.utils.data.DataLoader(
    datasets.MNIST('data', train=True, download=True,
                   transform=transforms.ToTensor()),
    batch_size=args.batch_size, shuffle=True)
#trainset_labeled = pickle.load(open("train_labeled.p", "rb"))
#validset = pickle.load(open("validation.p", "rb"))
#trainset_unlabeled = pickle.load(open("train_unlabeled.p", "rb"))

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

def ai(size,obj, name,u):
    para = nn.Parameter(torch.ones(1, size), requires_grad=True)
    obj.register_parameter(name,para)
    return para


shapes = zip(layer_sizes[:-1], layer_sizes[1:])  # shapes of linear layers


join = lambda l, u: torch.cat([l, u], 0)

#print(trainset_unlabeled.train_labels.size())

#train_loader = torch.utils.data.DataLoader(trainset_labeled , batch_size=64, shuffle=True, **kwargs)
#unlabeled_train_loader = torch.utils.data.DataLoader( trainset_unlabeled, batch_size=64, shuffle=True, **kwargs)

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
        self.ai = {}

    def split_input(self, h, labeled):
        if labeled:
            return h, h
        else:
            return None, h

    def g_gauss(self,z_c, u, size,layer):
        "gaussian denoising function proposed in the original paper"
        if not layer in self.ai:
            self.ai[layer] = {}
            #TODO : when to reinitialize
            self.ai[layer][1] = ai(size,self,"a1",u)
            self.ai[layer][2] = ai(size,self,"a2",u)
            self.ai[layer][3] = ai(size, self, "a3",u)
            self.ai[layer][4] = ai(size, self, "a4",u)
            self.ai[layer][5] = ai(size, self, "a5",u)
            self.ai[layer][6] = ai(size, self, "a6",u)
            self.ai[layer][7] = ai(size, self, "a7",u)
            self.ai[layer][8] = ai(size, self, "a8",u)
            self.ai[layer][9] = ai(size, self, "a9",u)
            self.ai[layer][10] = ai(size, self, "a10",u)

        temp1 = self.ai[layer][2].expand(u.size()[0], size) * u
        temp2 = temp1 + self.ai[layer][3].expand(u.size()[0], size)
        temp3 = self.ai[layer][4].expand(u.size()[0], size) * u

        mu = self.ai[layer][1].expand(u.size()[0], size) * nn.functional.sigmoid(temp2) + temp3 + self.ai[layer][5].expand(u.size()[0], size)
        v = self.ai[layer][6].expand(u.size()[0], size) * nn.functional.sigmoid(self.ai[layer][7].expand(u.size()[0], size) * u + self.ai[layer][8].expand(u.size()[0], size)) + self.ai[layer][9].expand(u.size()[0], size) * u + self.ai[layer][10].expand(u.size()[0], size)

        z_est = (z_c - mu) * v + mu
        return z_est

    def batch_norm(self, input):
        m = torch.mean(input)
        v = torch.var(input)

        ans = ((input - m.data[0]) / math.sqrt(v.data[0] + 1e-10)) + 0.3
        return (((input - m.data[0]) / math.sqrt(v.data[0] + 1e-10)) + 0.3),m,v

    def encoder_clean(self,h, noise_std, labeled):
        d = {'z': {}, 'm': {}, 'v': {}, 'h': {}}
        d['z'][0] = h
        d['z'][0] = h

        for l in xrange(1, L+1):
            # post activation from previos layer
            h_l_prev = d['h'][l - 1]

            # pre-activation and batch normalization
            z_pre = torch.mm(h_l_prev, self.weights['W'][l - 1])

            z_l,m_l, v_l = self.batch_norm(z_pre)

            if l == L:
            # use softmax activation in output layer
                softmax = torch.nn.Softmax()
                h = softmax(z)
            # TODO h = softmax(self.weights['gamma'][l - 1] * (z + self.weights["beta"][l - 1]))
            else:
                # use ReLU activation in hidden layers
                relU = torch.nn.ReLU()

                h_l = relU(z_l)  # TODO + self.weights["beta"][l - 1])

                d['z'][l] = z_l
                d['h'][l] = h_l

        return d['h'][L],d


    def encoder_noise(self,inputs, noise_std, labeled):
        noise = Variable(torch.mul(torch.randn(inputs.size()),noise_std))
        h = inputs + noise

        d = {'z': {}, 'm': {}, 'v': {}, 'h': {}}

        d['z'][0] = h
        d['z'][0] = h

        for l in xrange(1,L+1):

            #print("Layer " + str(l)+ ": "+ str(layer_sizes[l - 1])+ " -> "+ str(layer_sizes[l]))

            #post activation from previos layer
            h_l_prev = d['h'][l - 1]

            z_pre = torch.mm(h_l_prev, self.weights['W'][l - 1])  # pre-activation and batch normalization

            z_l,_,_  = self.batch_norm(z_pre)
            #TODO: BN for test is different

            relU = torch.nn.ReLU()

            h_l = relU(z_l) #TODO + self.weights["beta"][l - 1])

            d['z'][l] = z_l
            d['h'][l] = h_l

        return d['h'][L], d


    def decoder(self,h_clean,h_corr,  d_clean, d_corr):
        # Decoder
        z_est = {}
        z_est[L] = h_corr

        d_cost = Variable(torch.FloatTensor(L + 1), requires_grad=False) # to store the denoising cost of all layers

        for l in range(L, 0, -1): #TODO: last layer not run
            #print ("Layer ", l, ": ", layer_sizes[l + 1] if l + 1 < len(layer_sizes) else None, " -> ", layer_sizes[l], ", denoising cost: ", denoising_cost[l])

            if l == L:
                u_l =  z_est[L]
            else:
                u_l = torch.mm(z_est[l + 1], self.weights['V'][l])

            u_l = self.batch_norm(u)


            z_corr_encoder_l =  d_corr['z'][l]

            z_corr_denoised_l = self.g_gauss(z_corr_encoder_l,u_l,layer_sizes[l],l)

            m, v = d_clean['m'].get(l, 0), d_clean['v'].get(l, 1 - 1e-10)

            z_est_bn = (z_est[l] - m) / v # TODO: element wise

            z = z.detach()

            d_cost[l] = reconstruction_function(torch.sigmoid(z_est_bn), z)
            #d_cost[l] = torch.nn.functional.binary_cross_entropy(torch.sigmoid(z_est_bn), torch.sigmoid(z))

        return d_cost

    def forward(self, (x,target,labeled)):

        h_corr,d_corr = self.encoder_noise(x,0.3,labeled) #TODO: add noise
        h_clean,d_clean = self.encoder(x,0,labeled)

        d_cost = self.decoder(h_clean,h_corr, d_clean,d_corr)
        
        # calculate total unsupervised cost by adding the denoising cost of all layers
        #u_cost = torch.zeros(torch.size(d_cost[0]))

        #for ele in d_cost:
        #    u_cost += ele
        # u_cost = tf.add_n(d_cost)
        #print('xxxxxxxxx')
        #print("d_cost: " + str(d_cost))
        #u_cost = torch.sum(d_cost)
        #print('xxxxxxxxx')
        #print("u_cost: " + str(u_cost.size()))

        if labeled:
            s_cost = torch.nn.functional.cross_entropy(h_corr,target)  # supervised cost
            #return u_cost + s_cost
            return s_cost

        return u_cost

model  = VAE()

optimizer = optim.Adam(model.parameters(), lr=args.lr)#TODO: change to something


#for param in model.parameters():
#    print(type(param.data), param.size())

def train(epoch):
    model.train()
    train_loss = 0
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = Variable(data), Variable(target)

        data = data.view(-1,784) #TODO possible reason of bad accuracy
        optimizer.zero_grad()
        loss = model((data, target,True))
        loss.backward()
        train_loss += loss.data[0]
        optimizer.step()

        if batch_idx % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                       100. * batch_idx / len(train_loader), loss.data[0]))

    print('====> Epoch: {} Average loss: {:.4f}'.format(
        epoch, args.batch_size * train_loss / len(train_loader.dataset)))

    '''

    for batch_idx, (data, target) in enumerate(unlabeled_train_loader):
        data, target = Variable(data), Variable(target)
        data = data.view(-1, 784)  # TODO possible reason of bad accuracy
        optimizer.zero_grad()
        loss = model((data, target,False))
        loss.backward()
        optimizer.step()

        if batch_idx % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                       100. * batch_idx / len(train_loader), loss.data[0]))

    '''






for epoch in range(1, args.epochs + 1):
    train(epoch)
    #test(epoch, valid_loader)
