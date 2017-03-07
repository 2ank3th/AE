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

layer_sizes = [784, 700, 500, 250, 250, 250, 10] # default model
ensemble_one = [784, 600, 800, 500, 300, 100, 10] # ensemble_model 1
ensemble_two = [784, 780, 900, 600, 400, 50, 10] # ensemble_model 2
ensemble_three = [784, 720, 800, 700, 450, 80, 10] # ensemble_model 3

# Training settings
parser = argparse.ArgumentParser(description='PyTorch MNIST Example')
parser.add_argument('--batch-size', type=int, default=50, metavar='N',
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
parser.add_argument('--ensemble', type=bool, default=False, metavar='E',
                    help='enable ensemble mode (default: False)')
                    
args = parser.parse_args()


# hyperparameters that denote the importance of each layer
denoising_cost = [1000.0, 10.0, 0.10, 0.10, 0.10, 0.10, 0.10]


torch.manual_seed(args.seed)


print('loading data!')
trainset_labeled = pickle.load(open("train_labeled.p", "rb"))
#validset = pickle.load(open("validation.p", "rb"))
trainset_unlabeled = pickle.load(open("train_unlabeled.p", "rb"))

def bi(init, size, name, obj):
    para = nn.Parameter((init * torch.ones(size)).type(dtype))
    obj.register_parameter(name, para)
    return para



def wi(layer, shape, name, obj):
    para = nn.Parameter((torch.randn(shape[0],shape[1]) / math.sqrt(shape[0])).type(dtype), requires_grad=True)
    obj.register_parameter(name, para)
    return para


def ai(size, obj, name, u):
    para = nn.Parameter(torch.ones(1, size), requires_grad=False)
    obj.register_parameter(name,para)
    return para

shapes = zip(layer_sizes[:-1], layer_sizes[1:])  # shapes of linear layers
join = lambda l, u: torch.cat([l, u], 0)

#running_mean = [Variable(torch.zeros(layer_sizes[l]), requires_grad=False) for l in range(L)]
#running_var = [Variable(torch.ones(layer_sizes[l]), requires_grad=False) for l in range(L)]
#bn_assigns = [] # store updates to be made to average mean and variance

join = lambda l, u: torch.cat([l, u], 0)

train_loader = torch.utils.data.DataLoader(trainset_labeled , batch_size=args.batch_size, shuffle=True)
unlabeled_train_loader = torch.utils.data.DataLoader(trainset_unlabeled, batch_size=args.batch_size, shuffle=True)
#valid_loader = torch.utils.data.DataLoader(validset, batch_size=64, shuffle=True)

class VAE(nn.Module):
    def __init__(self, model_layer):
        super(VAE, self).__init__()
        
        shape = zip(model_layer[:-1], model_layer[1:])
        last_layer = len(model_layer)-1
        
        self.weights = {
           'W': [wi(i, s, "W", self) for i,s in enumerate(shape)],  # Encoder weights
           'V': [wi(i, s, "V", self) for i,s in enumerate(shape)],  # Decoder weights
           # batch normalization parameter to shift the normalized value
           'beta': [bi(0.0, model_layer[l+1], "beta", self) for l in range(last_layer)],
           # batch normalization parameter to scale the normalized value
           'gamma': [bi(1.0, model_layer[l+1], "beta", self) for l in range(last_layer)]
        }
        
        self.ai = {}

    def split_input(self, h, labeled):
        if labeled:
            return h, h
        else:
            return None, h

    def g_gauss(self, z_c, u, size, layer): #TODO: ERROR !! BAD FUNCTION !! fix to be dynamic size
        "gaussian denoising function proposed in the original paper"
        if not layer in self.ai:
            self.ai[layer] = {}
            
            self.ai[layer][1] = ai(size,self, "a1",0)
            self.ai[layer][2] = ai(size,self, "a2",1)
            self.ai[layer][3] = ai(size, self, "a3",0)
            self.ai[layer][4] = ai(size, self, "a4",0)
            self.ai[layer][5] = ai(size, self, "a5",0)
            self.ai[layer][6] = ai(size, self, "a6",0)
            self.ai[layer][7] = ai(size, self, "a7",1)
            self.ai[layer][8] = ai(size, self, "a8",0)
            self.ai[layer][9] = ai(size, self, "a9",0)
            self.ai[layer][10] = ai(size, self, "a10",0)

        temp1 = self.ai[layer][2].expand(u.size()[0], size) * u
        temp2 = temp1 + self.ai[layer][3].expand(u.size()[0], size)
        temp3 = self.ai[layer][4].expand(u.size()[0], size) * u

        mu = self.ai[layer][1].expand(u.size()[0], size) * nn.functional.sigmoid(temp2) + temp3 + self.ai[layer][5].expand(u.size()[0], size)
        v = self.ai[layer][6].expand(u.size()[0], size) * nn.functional.sigmoid(self.ai[layer][7].expand(u.size()[0], size) * u + self.ai[layer][8].expand(u.size()[0], size)) + self.ai[layer][9].expand(u.size()[0], size) * u + self.ai[layer][10].expand(u.size()[0], size)

        z_est = (z_c - mu) * v + mu
        return z_est

    def batch_norm(self, input):
        row = input.size()[0]
        column = input.size()[1]
        m = torch.mean(input, 0).expand(row, column)
        v = torch.var(input, 0).expand(row, column)
        return ((input - m) / (torch.sqrt(v + 1e-10)))


    def encoder(self, inputs, noise_std, labeled, net_model):
        h = inputs + Variable(torch.mul(torch.randn(inputs.size()),noise_std))
        d = {'z': {}, 'm': {}, 'v': {}, 'h': {}} # z = pre-activation, m = mean, v = variance, h = activated

        d['z'][0] = h
        d['h'][0] = h # no need to activate first layer
        
        last_layer = len(net_model)-1
        
        for l in range(1, last_layer+1):
            #print("Layer " + str(l)+ ": "+ str(layer_sizes[l - 1])+ " -> "+ str(layer_sizes[l]))

            z_pre = torch.mm(h, self.weights['W'][l-1])  # pre-activation
            z_pre_l, z_pre_u = self.split_input(z_pre,labeled)  # split labeled and unlabeled examples
 
            m, v = torch.mean(z_pre_u.data, 0), torch.var(z_pre_u.data, 0)
            d['m'][l], d['v'][l] = Variable(m), Variable(v)  # save mean and variance of unlabeled examples for decoding

            z = self.batch_norm(z_pre) + Variable(torch.mul(torch.randn(z_pre.size()),noise_std))        
            
            d['z'][l] = z
            
            b = self.weights["beta"][l-1]
            b = torch.t(b.resize(net_model[l], 1))
            b = b.expand(z.size()[0], net_model[l])
                
            g = self.weights['gamma'][l-1]
            g = torch.t(g.resize(net_model[l], 1))
            g = g.expand(z.size()[0], net_model[l])
            
            
            if l == last_layer:
                h = torch.nn.functional.softmax(g * (z + b)) # use softmax activation in output layer
            else:
                h = torch.nn.functional.relu(z + b) # use ReLU activation in hidden layers
     
            d['h'][l] = h

        return h, d


    def decoder(self, h_clean, h_corr, d_clean, d_corr, net_model):
        last_layer = len(net_model)-1
        z_est = {}
        d_cost = Variable(torch.zeros(last_layer+1), requires_grad=False) # to store the denoising cost of all layers
          
        for l in range(last_layer, 0, -1):
            #print ("Layer ", l, ": ", layer_sizes[l + 1] if l + 1 < len(layer_sizes) else None, " -> ", layer_sizes[l], ", denoising cost: ", denoising_cost[l])
            
            z, z_c = d_clean['z'][l], d_corr['z'][l]
            m, v = d_clean['m'].get(l, 0), d_clean['v'].get(l, 1-1e-10)

            if l == last_layer:
                u =  h_corr
            else:
                u = torch.mm(z_est[l+1], torch.t(self.weights['V'][l]))

            u = self.batch_norm(u)
            z_est[l] = self.g_gauss(z_c, u, net_model[l], l)
            
            size = z_est[l].size()
            m = m.expand(size[0], net_model[l])
            v = v.expand(size[0], net_model[l])

            z_est_bn = (z_est[l] - m) / torch.sqrt(v)
            
            #TODO: reconstruction cost should be L2 norm          
            d_cost[l] = (torch.mean(torch.sum(torch.pow((z_est_bn - z),2),1)) / net_model[l]) * denoising_cost[l]

        return d_cost

    def forward(self, (x,target,labeled), net_model):
        input_images = x.view(-1,784)
        
        h_corr,d_corr = self.encoder(input_images, 0.3, labeled, net_model)
        h_clean,d_clean = self.encoder(input_images,0, labeled, net_model)            
        
        d_cost = self.decoder(h_clean, h_corr, d_clean, d_corr, net_model)
        u_cost = torch.sum(d_cost)
       
        if labeled:
            s_cost = -torch.mean(torch.sum(target*torch.log(h_corr), 1)) # supervised cost
            return u_cost + s_cost
        
        return u_cost
        
model = VAE(layer_sizes)
model_en_one = VAE(ensemble_one)
model_en_two = VAE(ensemble_two)
model_en_three = VAE(ensemble_three)


optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=0.99)
optimizer_en_one = optim.Adam(model_en_one.parameters(), lr=args.lr, weight_decay=0.99)
optimizer_en_two = optim.Adam(model_en_two.parameters(), lr=args.lr, weight_decay=0.99)
optimizer_en_three = optim.Adam(model_en_three.parameters(), lr=args.lr, weight_decay=0.99)

def target_vector(num, net_model): # convert target values into vectors
    last_layer = len(net_model)-1
    t_vec = torch.zeros(num.size()[0], net_model[last_layer])
    
    for i in range(t_vec.size()[0]):
        val = num[i]
        t_vec[i][val] = 1
    return t_vec

def train(epoch):

    if args.ensemble == True:
        print("---ENSEMBLE MODE---")
        model_en_one.train()
        model_en_two.train()
        model_en_three.train()
        

        for batch_idx, (data, target) in enumerate(train_loader):
            target = target_vector(target, layer_sizes)
            data, target = Variable(data, requires_grad=False), Variable(target, requires_grad=False)
            
            # Ensemble Model 1
            optimizer_en_one.zero_grad()
            loss_one = model_en_one((data, target, True), ensemble_one)
            loss_one.backward()
            optimizer_en_one.step()
            
            # Ensemble Model 2
            optimizer_en_two.zero_grad()
            loss_two = model_en_two((data, target, True), ensemble_two)
            loss_two.backward()
            optimizer_en_two.step()
                    
            # Ensemble Model 3
            optimizer_en_three.zero_grad()
            loss_three = model_en_three((data, target, True), ensemble_three)
            loss_three.backward()
            optimizer_en_three.step()
            
            if batch_idx % args.log_interval == 0:
                print('Train Epoch: {} [{}/{} ({:.0f}%)]\tAverage Loss: {:.6f}'.format(
                    epoch, batch_idx * len(data), len(train_loader.dataset),
                           100. * batch_idx / len(train_loader), 
                           (loss_one.data[0] + loss_two.data[0] + loss_three.data[0]) / 3))

        for batch_idx, (data, target) in enumerate(unlabeled_train_loader):
            target = target_vector(target, layer_sizes)    
            data, target = Variable(data, requires_grad=False), Variable(target, requires_grad=False)

            # Ensemble Model 1
            optimizer_en_one.zero_grad()
            loss_one = model_en_one((data, target, False), ensemble_one)
            loss_one.backward()
            optimizer_en_one.step()
            
            # Ensemble Model 2
            optimizer_en_two.zero_grad()
            loss_two = model_en_two((data, target, False), ensemble_two)
            loss_two.backward()
            optimizer_en_two.step()
                    
            # Ensemble Model 3
            optimizer_en_three.zero_grad()
            loss_three = model_en_three((data, target, False), ensemble_three)
            loss_three.backward()
            optimizer_en_three.step()
            
            
            if batch_idx % args.log_interval == 0:
                print('Train Epoch: {} [{}/{} ({:.0f}%)]\tAverage Loss: {:.6f}'.format(
                    epoch, batch_idx * len(data), len(train_loader.dataset),
                           100. * batch_idx / len(train_loader), 
                           (loss_one.data[0] + loss_two.data[0] + loss_three.data[0]) / 3))
        
        else:
            model.train()
    
            for batch_idx, (data, target) in enumerate(train_loader):
                target = target_vector(target, layer_sizes)
                data, target = Variable(data, requires_grad=False), Variable(target, requires_grad=False)
                optimizer.zero_grad()
                loss = model((data, target, True), layer_sizes)
                loss.backward()
                optimizer.step()
        
                if batch_idx % args.log_interval == 0:
                    print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                        epoch, batch_idx * len(data), len(train_loader.dataset),
                               100. * batch_idx / len(train_loader), loss.data[0]))

            for batch_idx, (data, target) in enumerate(unlabeled_train_loader):
                target = target_vector(target, layer_sizes)    
                data, target = Variable(data, requires_grad=False), Variable(target, requires_grad=False)
                optimizer.zero_grad()
                loss = model((data, target,False), layer_sizes)
                loss.backward()
                optimizer.step()

                if batch_idx % args.log_interval == 0:
                    print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                        epoch, batch_idx * len(data), len(train_loader.dataset),
                               100. * batch_idx / len(train_loader), loss.data[0]))

for epoch in range(1, args.epochs + 1):
    train(epoch)
    #test(epoch, valid_loader)
