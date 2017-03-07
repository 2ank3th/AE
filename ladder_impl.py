from __future__ import print_function

import torch
from torch.autograd import Variable
import argparse
import cPickle as pickle

import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms


dtype = torch.FloatTensor
layer_sizes = [784, 1000, 500, 250, 250, 250, 10]
L = len(layer_sizes) - 1

# Training settings
parser = argparse.ArgumentParser(description='PyTorch MNIST Example')
parser.add_argument('--batch-size', type=int, default=64, metavar='N',
                    help='input batch size for training (default: 64)')
parser.add_argument('--test-batch-size', type=int, default=64, metavar='N',
                    help='input batch size for testing (default: 1000)')
parser.add_argument('--epochs', type=int, default=50, metavar='N',
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
denoising_cost = [10.0, 1.0, 0.10, 0.10, 0.10, 0.10, 0.10]


torch.manual_seed(args.seed)


print('loading data!')
train_loader = torch.utils.data.DataLoader(
    datasets.MNIST('data', train=True, download=True,
                   transform=transforms.ToTensor()),
    batch_size=args.batch_size, shuffle=True)
trainset_labeled = pickle.load(open("train_labeled.p", "rb"))
validset = pickle.load(open("validation.p", "rb"))
trainset_unlabeled = pickle.load(open("train_unlabeled.p", "rb"))

def bi(layer, size, name,obj):
    #temp = torch.randn(size).type(dtype)
    #tv = Variable(temp)
    #tv.register_parameter(
    para = nn.Parameter(torch.randn(1,size).type(dtype))
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

#print(trainset_unlabeled.train_labels.size())

train_loader = torch.utils.data.DataLoader(trainset_labeled , batch_size=64, shuffle=True)
unlabeled_train_loader = torch.utils.data.DataLoader( trainset_unlabeled, batch_size=64, shuffle=True)
valid_loader = torch.utils.data.DataLoader(validset, batch_size=64, shuffle=True)

reconstruction_function = nn.BCELoss()
reconstruction_function.size_average = False

class Denoise(nn.Module):
    def __init__(self,size):
        super(Denoise, self).__init__()
        self.a1 = nn.Parameter(torch.ones(1, size), requires_grad=True)
        self.a2 = nn.Parameter(torch.ones(1, size), requires_grad=True)
        self.a3 = nn.Parameter(torch.ones(1, size), requires_grad=True)
        self.a4 = nn.Parameter(torch.ones(1, size), requires_grad=True)
        self.a5 = nn.Parameter(torch.ones(1, size), requires_grad=True)
        self.a6 = nn.Parameter(torch.ones(1, size), requires_grad=True)
        self.a7 = nn.Parameter(torch.ones(1, size), requires_grad=True)
        self.a8 = nn.Parameter(torch.ones(1, size), requires_grad=True)
        self.a9 = nn.Parameter(torch.ones(1, size), requires_grad=True)
        self.a10 = nn.Parameter(torch.ones(1, size), requires_grad=True)

    def denoise_function(self,z_c, u, size):

        temp1 = self.a2.expand(u.size()[0], size) * u
        temp2 = temp1 + self.a3.expand(u.size()[0], size)
        temp3 = self.a4.expand(u.size()[0], size) * u

        mu = self.a1.expand(u.size()[0], size) * nn.functional.sigmoid(temp2) + temp3 + self.a5.expand(u.size()[0], size)
        v = self.a6.expand(u.size()[0], size) * nn.functional.sigmoid(self.a7.expand(u.size()[0], size) * u + self.a8.expand(u.size()[0], size)) + self.a9.expand(u.size()[0], size) * u + self.a10.expand(u.size()[0], size)

        z_est = (z_c - mu) * v + mu
        return z_est

    def forward(self, u,z_corr, m,v,layer):
        z_corr_denoised_l = self.denoise_function(z_corr, u, layer_sizes[layer])

        z_est = (z_corr_denoised_l - m) / torch.pow(torch.add(v, 0.000001) ,0.5)

        return z_est


class VAE(nn.Module):
    def __init__(self):
        super(VAE, self).__init__()
              
        self.weights = {'W': [wi(i,s, "W",self) for i,s in enumerate(shapes)],  # Encoder weights
           'V': [wi(i,s[::-1], "V",self) for i,s in enumerate(shapes)],  # Decoder weights
           # batch normalization parameter to shift the normalized value
           'beta': [bi(l+1, layer_sizes[l+1], "beta",self) for l in range(L)],
           # batch normalization parameter to scale the normalized value
           'gamma': [bi(l+1, layer_sizes[l+1], "beta",self) for l in range(L)]}

        self.a = {}
        self.optim = {}
        self.running_var = [Variable(torch.ones(1,l), requires_grad=False) for l in layer_sizes[1:]]
        self.running_mean = [Variable(torch.zeros(1, l), requires_grad=False) for l in layer_sizes[1:]]

        for i in range(L+1):
            self.a[i] = Denoise(layer_sizes[i])
            self.optim[i] = optim.Adam(self.a[i].parameters(), lr=args.lr)

    def split_input(self, h, labeled):
        if labeled:
            return h, h
        else:
            return None, h

    def batch_norm(self, input, affine_flag):
        row = input.size()[0]
        column = input.size()[1]
        m = torch.mean(input, 0) #.expand(row, column)
        v = torch.var(input, 0)#.expand(row, column)
        bn =  nn.BatchNorm1d(column,affine = affine_flag)
        return bn(input),m,v

    def encoder_clean(self,h,labeled):
        d = {'z': {}, 'm': {}, 'v': {}, 'h': {}}
        d['z'][0] = h
        d['h'][0] = h

        for l in xrange(1, L+1):
            # post activation from previos layer
            h_l_prev = d['h'][l - 1]


            # pre-activation and batch normalization
            z_pre = torch.mm(h_l_prev, self.weights['W'][l - 1])

            z_l, m_l, v_l = self.batch_norm(z_pre, False)


            d['m'][l] = m_l
            d['v'][l] = v_l

            if l == L:
            # use softmax activation in output layer
                softmax = torch.nn.Softmax()
                h_l = softmax(z_l)
                d['z'][l] = z_l
                d['h'][l] = h_l
            else:
                # use ReLU activation in hidden layers
                relU = torch.nn.ReLU()

                h_l = relU(z_l)

                d['z'][l] = z_l
                d['h'][l] = h_l

        return d['h'][L],d


    def encoder_noise(self,inputs, noise_std, labeled):
        noise = Variable(torch.mul(torch.randn(inputs.size()),noise_std))

        h = inputs + noise

        d = {'z': {}, 'm': {}, 'v': {}, 'h': {}}

        d['z'][0] = h
        d['h'][0] = h

        for l in xrange(1,L+1):

            #print("Layer " + str(l)+ ": "+ str(layer_sizes[l - 1])+ " -> "+ str(layer_sizes[l]))

            #post activation from previos layer
            h_l_prev = d['h'][l - 1]
            z_pre = torch.mm(h_l_prev, self.weights['W'][l - 1])
            z_pre =  z_pre + Variable(torch.mul(torch.randn(z_pre.size()),noise_std)) # pre-activation and batch normalization

            z_l,_,_  = self.batch_norm(z_pre, False)

            relU = torch.nn.ReLU()

            h_l = relU(z_l)

            d['z'][l] = z_l
            d['h'][l] = h_l

        return d['h'][L], d

    def reconstruction_function_n(self, z_recon,z_input):
        return torch.norm(z_recon - z_input, 2, 1)


    def decoder(self,h_clean,h_corr,  d_clean, d_corr):
        # Decoder
        z_est = {}
        z_est[L] = h_corr

        d_cost = Variable(torch.FloatTensor(L + 1), requires_grad=False)

        for l in range(L, 0, -1):

            if l == L:
                u_l =  z_est[L]
            else:
                u_l = torch.mm(z_est[l + 1], self.weights['V'][l])

            u_l,_,_ = self.batch_norm(u_l, False)


            z_corr_encoder_l=d_corr['z'][l]

            #z_corr_denoised_l = self.g_gauss(z_corr_encoder_l,u_l,layer_sizes[l],l)

            m, v = d_clean['m'][l], d_clean['v'][l]

            z = d_clean['z'][l]

            z = z.detach()

            z_est[l] = self.a[l](u_l,z_corr_encoder_l,m.expand(u_l.size()[0],u_l.size()[1]),v.expand(u_l.size()[0],u_l.size()[1]),l)
            self.optim[l].zero_grad()

            recon_loss = torch.sum(self.reconstruction_function_n(z_est[l], z))
            recon_loss.backward(retain_variables=True)
            self.optim[l].step()

            d_cost[l] = torch.mul(recon_loss,denoising_cost[l])

        return d_cost

    def forward(self, x,target,labeled,test):

        h_corr,d_corr = self.encoder_noise(x,0.2,labeled) #TODO: add noise
        h_clean,d_clean = self.encoder_clean(x,labeled)

        if test:
            s_cost = torch.nn.functional.cross_entropy(h_corr, target)  # supervised cost
            return h_clean,s_cost

        d_cost = self.decoder(h_clean,h_corr, d_clean,d_corr)

        u_cost = torch.sum(d_cost)


        if labeled :
            s_cost = torch.nn.functional.cross_entropy(h_corr,target)  # supervised cost
            return h_corr,(s_cost + u_cost)


        return h_corr,u_cost

model  = VAE()

optimizer = optim.Adam(model.parameters(), lr=args.lr)



#for key in model.a:
#    for param in model.a[key].parameters():
#        print(type(param.data), param.size())

def train(epoch):
    model.train()
    train_loss = 0


    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = Variable(data), Variable(target)

        data = data.view(-1,784) #TODO possible reason of bad accuracy

        output,loss = model(data, target,True,False)
        optimizer.zero_grad()
        loss.backward()
        train_loss += loss.data[0]
        optimizer.step()

        if batch_idx % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                       100. * batch_idx / len(train_loader), loss.data[0]))

    print('====> Epoch: {} Average loss: {:.4f}'.format(
        epoch, args.batch_size * train_loss / len(train_loader.dataset)))



    for batch_idx, (data, target) in enumerate(unlabeled_train_loader):
        data, target = Variable(data), Variable(target)
        data = data.view(-1, 784)  # TODO possible reason of bad accuracy

        output,loss = model(data, target,False, False )
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if batch_idx % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(unlabeled_train_loader.dataset),
                       100. * batch_idx / len(unlabeled_train_loader), loss.data[0]))


    print("epoch {} completed",epoch)





def test(epoch, valid_loader):

    model.eval()
    test_loss = 0
    correct = 0

    for data, target in valid_loader:
        data, target = Variable(data, volatile=True), Variable(target)
        data = data.view(-1, 784)
        output,loss = model(data,target, True, True)
        test_loss += F.nll_loss(output, target).data[0]
        pred = output.data.max(1)[1] # get the index of the max log-probability
        correct += pred.eq(target.data).cpu().sum()

    test_loss /= len(valid_loader) # loss function already averages over batch size
    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(valid_loader.dataset),
        100. * correct / len(valid_loader.dataset)))


for epoch in range(1, args.epochs + 1):
    train(epoch)
    test(epoch, valid_loader)
