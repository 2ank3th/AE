from __future__ import print_function

import torch
from torch.autograd import Variable
import argparse
import cPickle as pickle
import os
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from DAE import DAE

layer_sizes = [784, 1000, 500, 250, 250, 250, 10]


# Training settings
parser = argparse.ArgumentParser(description='PyTorch MNIST Example')
parser.add_argument('--batch-size', type=int, default=64, metavar='N',
                    help='input batch size for training (default: 64)')
parser.add_argument('--test-batch-size', type=int, default=64, metavar='N',
                    help='input batch size for testing (default: 1000)')
parser.add_argument('--epochs', type=int, default=25, metavar='N',
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




torch.manual_seed(args.seed)


print('loading data!')
train_loader = torch.utils.data.DataLoader(
    datasets.MNIST('data', train=True, download=True,
                   transform=transforms.ToTensor()),
    batch_size=args.batch_size, shuffle=True)
trainset_labeled = pickle.load(open("train_labeled.p", "rb"))
validset = pickle.load(open("validation.p", "rb"))
trainset_unlabeled = pickle.load(open("train_unlabeled.p", "rb"))




join = lambda l, u: torch.cat([l, u], 0)

#print(trainset_unlabeled.train_labels.size())

train_loader = torch.utils.data.DataLoader(trainset_labeled , batch_size=64, shuffle=True)
unlabeled_train_loader = torch.utils.data.DataLoader( trainset_unlabeled, batch_size=64, shuffle=True)
valid_loader = torch.utils.data.DataLoader(validset, batch_size=64, shuffle=True)




model  = DAE()

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

        output,loss = model(data, target,True,False, False)
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


    '''
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
    '''





def test(epoch, valid_loader):

    model.eval()
    test_loss = 0
    correct = 0

    for data, target in valid_loader:
        data, target = Variable(data, volatile=True), Variable(target)
        data = data.view(-1, 784)
        output,loss = model(data,target, True, True, False)
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

torch.save(model.state_dict(),  os.getcwd() + "/model.pth")
