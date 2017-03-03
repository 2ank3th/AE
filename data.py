
# coding: utf-8

# In[135]:

from __future__ import print_function
import pickle 
import numpy as np
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.autograd import Variable


# In[136]:

from sub import subMNIST       # testing the subclass of MNIST dataset


# # Split Data

# In[137]:

transform=transforms.Compose([transforms.ToTensor(),
                              transforms.Normalize((0.1307,), (0.3081,))
                             ])


# In[138]:

trainset_original = datasets.MNIST('../data', train=True, download=True,
                                  transform=transform)


# In[139]:

train_label_index = []
valid_label_index = []
for i in range(10):
    train_label_list = trainset_original.train_labels.numpy()
    label_index = np.where(train_label_list == i)[0]
    label_subindex = list(label_index[:300])
    valid_subindex = list(label_index[300: 1000 + 300])
    train_label_index += label_subindex
    valid_label_index += valid_subindex


# ## Train Set

# In[140]:

trainset_np = trainset_original.train_data.numpy()
trainset_label_np = trainset_original.train_labels.numpy()
train_data_sub = torch.from_numpy(trainset_np[train_label_index])
train_labels_sub = torch.from_numpy(trainset_label_np[train_label_index])


# In[141]:

trainset_new = subMNIST(root='./data', train=True, download=True, transform=transform, k=3000)
trainset_new.train_data = train_data_sub.clone()
trainset_new.train_labels = train_labels_sub.clone()


# In[142]:

trainset_new.train_data.size()


# In[143]:

pickle.dump(trainset_new, open("train_labeled.p", "wb" ))


# ## Validation Set

# In[144]:

validset_np = trainset_original.train_data.numpy()
validset_label_np = trainset_original.train_labels.numpy()
valid_data_sub = torch.from_numpy(validset_np[valid_label_index])
valid_labels_sub = torch.from_numpy(validset_label_np[valid_label_index])


# In[145]:

validset = subMNIST(root='./data', train=False, download=True, transform=transform, k=10000)
validset.test_data = valid_data_sub.clone()
validset.test_labels = valid_labels_sub.clone()


# In[146]:

validset.test_data.size()


# In[147]:

pickle.dump(validset, open("validation.p", "wb" ))


# ## Unlabeled Data

# In[148]:

train_unlabel_index = []
for i in range(60000):
    if i in train_label_index or i in valid_label_index:
        pass
    else:
        train_unlabel_index.append(i)


# In[149]:

trainset_np = trainset_original.train_data.numpy()
trainset_label_np = trainset_original.train_labels.numpy()
train_data_sub_unl = torch.from_numpy(trainset_np[train_unlabel_index])
train_labels_sub_unl = torch.from_numpy(trainset_label_np[train_unlabel_index])


# In[150]:

trainset_new_unl = subMNIST(root='./data', train=True, download=True, transform=transform, k=47000)
trainset_new_unl.train_data = train_data_sub_unl.clone()
trainset_new_unl.train_labels = [-1]*47000      # Unlabeled!!


# In[151]:

trainset_new_unl.train_data.size()


# In[152]:

trainset_new_unl.train_labels


# In[153]:

pickle.dump(trainset_new_unl, open("train_unlabeled.p", "wb" ))


# # Train Model

# In[154]:

# train_loader = torch.utils.data.DataLoader(trainset_new, batch_size=64, shuffle=True)
# valid_loader = torch.utils.data.DataLoader(validset, batch_size=64, shuffle=True)


# In[155]:

trainset_imoprt = pickle.load(open("train_labeled.p", "rb"))
validset_import = pickle.load(open("validation.p", "rb"))


# In[156]:

train_loader = torch.utils.data.DataLoader(trainset_imoprt, batch_size=64, shuffle=True)
valid_loader = torch.utils.data.DataLoader(validset_import, batch_size=64, shuffle=True)


# In[157]:

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(320, 50)
        self.fc2 = nn.Linear(50, 10)

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        x = x.view(-1, 320)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = F.relu(self.fc2(x))
        return F.log_softmax(x)

model = Net()


# In[158]:

optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.5)


# In[159]:

# CPU only training
def train(epoch):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):

        data, target = Variable(data), Variable(target)
        optimizer.zero_grad()
        output = model(data)
        loss = F.nll_loss(output, target)
        loss.backward()
        optimizer.step()
        if batch_idx % 10 == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.data[0]))

def test(epoch, valid_loader):
    model.eval()
    test_loss = 0
    correct = 0
    for data, target in valid_loader:

        data, target = Variable(data, volatile=True), Variable(target)
        output = model(data)
        test_loss += F.nll_loss(output, target).data[0]
        pred = output.data.max(1)[1] # get the index of the max log-probability
        correct += pred.eq(target.data).cpu().sum()

    test_loss /= len(valid_loader) # loss function already averages over batch size
    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(valid_loader.dataset),
        100. * correct / len(valid_loader.dataset)))


# In[160]:

for epoch in range(1, 11):
    train(epoch)
    test(epoch, valid_loader)


# # Create Sample Submission

# In[161]:

testset = datasets.MNIST('../data', train=False, transform=transform)


# In[162]:

pickle.dump(testset, open("test.p", "wb" ))


# In[163]:

test_loader = torch.utils.data.DataLoader(testset,batch_size=64, shuffle=False)


# ## Test Accuuracy

# In[164]:

test(1, test_loader)


# In[165]:

label_predict = np.array([])
model.eval()
for data, target in test_loader:
    data, target = Variable(data, volatile=True), Variable(target)
    output = model(data)
    temp = output.data.max(1)[1].numpy().reshape(-1)
    label_predict = np.concatenate((label_predict, temp))


# In[166]:

label_predict


# In[167]:

label_true = test_loader.dataset.test_labels.numpy()


# In[168]:

diff_array = label_true - label_predict


# In[169]:

len(np.where(diff_array != 0)[0])


# In[170]:

import pandas as pd
true_label = pd.DataFrame(label_true, columns=['label'])
true_label.reset_index(inplace=True)
true_label.rename(columns={'index': 'ID'}, inplace=True)


# In[171]:

true_label.head()


# In[172]:

predict_label = pd.DataFrame(label_predict, columns=['label'], dtype=int)
predict_label.reset_index(inplace=True)
predict_label.rename(columns={'index': 'ID'}, inplace=True)


# In[173]:

predict_label.head()


# In[174]:

predict_label.to_csv('sample_submission.csv', index=False)
true_label.to_csv('true_label.csv', index=False)

