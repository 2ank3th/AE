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
<<<<<<< Updated upstream
parser.add_argument('--lr', type=float, default=0.01, metavar='LR',
=======
parser.add_argument('--lr', type=float, default=0.02, metavar='LR',
>>>>>>> Stashed changes
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

<<<<<<< Updated upstream
def bi(layer, size, name,obj):
    #temp = torch.randn(size).type(dtype)
    #tv = Variable(temp)
    #tv.register_parameter(
    para = nn.Parameter(torch.randn(1,size).type(dtype))
    obj.register_parameter(str(layer)+name+str(size),para)
=======
def bi(init, size, name, obj):
    para = nn.Parameter((init * torch.ones(size)).type(dtype))
    obj.register_parameter(name, para)
>>>>>>> Stashed changes
    return para


def wi(layer,shape, name,obj):
    para = nn.Parameter((torch.randn(shape[0],shape[1]) / math.sqrt(shape[0])).type(dtype), requires_grad=True)
    obj.register_parameter(name, para)
    return para

<<<<<<< Updated upstream
=======
def ai(size,obj, name,u):
    para = nn.Parameter(torch.ones(1, size), requires_grad=False)
    obj.register_parameter(name,para)
    return para

>>>>>>> Stashed changes

shapes = zip(layer_sizes[:-1], layer_sizes[1:])  # shapes of linear layers
join = lambda l, u: torch.cat([l, u], 0)

#running_mean = [Variable(torch.zeros(layer_sizes[l]), requires_grad=False) for l in range(L)]
#running_var = [Variable(torch.ones(layer_sizes[l]), requires_grad=False) for l in range(L)]
#bn_assigns = [] # store updates to be made to average mean and variance

#print(trainset_unlabeled.train_labels.size())
<<<<<<< Updated upstream

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


=======
train_loader = torch.utils.data.DataLoader(trainset_labeled , batch_size=args.batch_size, shuffle=True)
unlabeled_train_loader = torch.utils.data.DataLoader( trainset_unlabeled, batch_size=args.batch_size, shuffle=True)

#valid_loader = torch.utils.data.DataLoader(validset, batch_size=64, shuffle=True)

>>>>>>> Stashed changes
class VAE(nn.Module):
    def __init__(self):
        super(VAE, self).__init__()
        
        self.weights = {
           'W': [wi(i, s, "W", self) for i,s in enumerate(shapes)],  # Encoder weights
           'V': [wi(i, s, "V", self) for i,s in enumerate(shapes)],  # Decoder weights
           # batch normalization parameter to shift the normalized value
<<<<<<< Updated upstream
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
=======
           'beta': [bi(0.0, layer_sizes[l+1], "beta", self) for l in range(L)],
           # batch normalization parameter to scale the normalized value
           'gamma': [bi(1.0, layer_sizes[l+1], "beta", self) for l in range(L)]
        }
        self.ai = {}
>>>>>>> Stashed changes

    def split_input(self, h, labeled):
        if labeled:
            return h, h
        else:
            return None, h

<<<<<<< Updated upstream
    def batch_norm(self, input, affine_flag):
        row = input.size()[0]
        column = input.size()[1]
        m = torch.mean(input, 0) #.expand(row, column)
        v = torch.var(input, 0)#.expand(row, column)
        bn =  nn.BatchNorm1d(column,affine = affine_flag)
        return bn(input),m,v
=======
    def g_gauss(self, z_c, u, size, layer):
        "gaussian denoising function proposed in the original paper"
        if not layer in self.ai:
            self.ai[layer] = {}
            #TODO : when to reinitialize
            self.ai[layer][1] = ai(size,self,"a1",0)
            self.ai[layer][2] = ai(size,self,"a2",1)
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
>>>>>>> Stashed changes

    def encoder_clean(self,h,labeled):
        d = {'z': {}, 'm': {}, 'v': {}, 'h': {}}
        d['z'][0] = h
        d['h'][0] = h

<<<<<<< Updated upstream
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
=======
    def batch_norm(self, input):
        row = input.size()[0]
        column = input.size()[1]
        m = torch.mean(input, 0).expand(row, column)
        v = torch.var(input, 0).expand(row, column)
        return ((input - m) / (torch.sqrt(v + 1e-10)))
        #return #((input - m.data[0]) / (math.sqrt(v.data[0]) + 1e-10))

    #def update_batch_norm(self, input, layer):
    #    m = torch.mean(input, 0)
    #    v = torch.var(input, 0)
    #    assign_mean = running_mean[l-1].assign(m)
    #    assign_var = running_var[l-1].assign(v)
    #    bn_assigns.append(ewma.apply([running_mean[l-1], running_var[l-1]]))
    #    return (input - m) / (torch.sqrt(v + 1e-10))

    def encoder(self, inputs, noise_std, labeled):
        h = inputs + Variable(torch.mul(torch.randn(inputs.size()),noise_std))
        d = {'z': {}, 'm': {}, 'v': {}, 'h': {}} # z = pre-activation, m = mean, v = variance, h = activated

        d['z'][0] = h
        d['h'][0] = h # no need to activate first layer
        
        for l in xrange(1,L+1):
            #print("Layer " + str(l)+ ": "+ str(layer_sizes[l - 1])+ " -> "+ str(layer_sizes[l]))

            z_pre = torch.mm(h, self.weights['W'][l-1])  # pre-activation
            z_pre_l, z_pre_u = self.split_input(z_pre,labeled)  # split labeled and unlabeled examples
            
            #TODO why only for unlabbeled if not labeled:
            #if noise_std == 0:
            m, v = torch.mean(z_pre_u.data, 0), torch.var(z_pre_u.data, 0)#torch.mean(z_pre_u.data), torch.var(z_pre_u.data)
            d['m'][l], d['v'][l] = Variable(m), Variable(v)  # save mean and variance of unlabeled examples for decoding
>>>>>>> Stashed changes

        h = inputs + noise

<<<<<<< Updated upstream
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
=======
            z = self.batch_norm(z_pre) + Variable(torch.mul(torch.randn(z_pre.size()),noise_std))
            
            
            d['z'][l] = z
            #TODO: BN for test is different
            
            b = self.weights["beta"][l-1]
            b = torch.t(b.resize(layer_sizes[l], 1))
            b = b.expand(z.size()[0], layer_sizes[l])
                
            g = self.weights['gamma'][l-1]
            g = torch.t(g.resize(layer_sizes[l], 1))
            g = g.expand(z.size()[0], layer_sizes[l])
            
            if l == L:
                # use softmax activation in output layer
                h = torch.nn.functional.softmax(g * (z + b))# + self.weights["beta"][l - 1])
                #TODO h = softmax(self.weights['gamma'][l - 1] * (z + self.weights["beta"][l - 1]))
            else:
                # use ReLU activation in hidden layers
                h = torch.nn.functional.relu(z + b) # + Variable(self.weights["beta"][l - 1].data)
     
            d['h'][l] = h #self.split_input(h,labeled)
>>>>>>> Stashed changes

            h_l = relU(z_l)

            d['z'][l] = z_l
            d['h'][l] = h_l

        return d['h'][L], d

    def reconstruction_function_n(self, z_recon,z_input):
        return torch.norm(z_recon - z_input, 2, 1)


    def decoder(self, h_clean, h_corr, d_clean, d_corr):
        # Decoder
        z_est = {}
<<<<<<< Updated upstream
        z_est[L] = h_corr

        d_cost = Variable(torch.FloatTensor(L + 1), requires_grad=False)

        for l in range(L, 0, -1):

=======
        d_cost = Variable(torch.zeros(L + 1), requires_grad=False) # to store the denoising cost of all layers
        
        for l in range(L, 0, -1): #TODO: last layer not run
            #print ("Layer ", l, ": ", layer_sizes[l + 1] if l + 1 < len(layer_sizes) else None, " -> ", layer_sizes[l], ", denoising cost: ", denoising_cost[l])

            z, z_c = d_clean['z'][l], d_corr['z'][l]
            m, v = d_clean['m'].get(l, 0), d_clean['v'].get(l, 1-1e-10)

>>>>>>> Stashed changes
            if l == L:
                u_l =  z_est[L]
            else:
<<<<<<< Updated upstream
                u_l = torch.mm(z_est[l + 1], self.weights['V'][l])

            u_l,_,_ = self.batch_norm(u_l, False)

=======
                u = torch.mm(z_est[l+1], torch.t(self.weights['V'][l]))
                #u = torch.mm(z_est[l+1], torch.t(self.weights['W'][l]))
                # TODO: CHECK ! u = torch.mm(z_est[l + 1], self.weights['V'][l])
                #u = tf.matmul(z_est[l+1], weights['V'][l])
>>>>>>> Stashed changes

            z_corr_encoder_l=d_corr['z'][l]

            #z_corr_denoised_l = self.g_gauss(z_corr_encoder_l,u_l,layer_sizes[l],l)

<<<<<<< Updated upstream
            m, v = d_clean['m'][l], d_clean['v'][l]

            z = d_clean['z'][l]
=======
            z_est[l] = self.g_gauss(z_c, u, layer_sizes[l], l)
            
            size = z_est[l].size()
            m = m.expand(size[0], layer_sizes[l])
            v = v.expand(size[0], layer_sizes[l])
>>>>>>> Stashed changes

            z_est_bn = (z_est[l] - m) / torch.sqrt(v)
            z = z.detach()
<<<<<<< Updated upstream

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


=======
            
            #TODO: reconstruction cost should be L2 norm          
            d_cost[l] = (torch.mean(torch.sum(torch.pow((z_est_bn - z),2),1)) / layer_sizes[l]) * denoising_cost[l]
            #torch.sum(torch.sqrt(torch.pow((z_est_bn - z),2)))
            #d_cost[l] = torch.nn.functional.binary_cross_entropy(torch.sigmoid(z_est_bn), torch.sigmoid(z))
            
        return d_cost

    def forward(self, (x,target,labeled)):
        input_images = x.view(-1,784)
        
        h_corr,d_corr = self.encoder(input_images, 0.3, labeled) #TODO: add noise / h_corr = P(y_corr | x)
        h_clean,d_clean = self.encoder(input_images,0, labeled)
        d_cost = self.decoder(h_clean, h_corr, d_clean, d_corr)
        
        # calculate total unsupervised cost by adding the denoising cost of all layers
        #u_cost = torch.zeros(torch.size(d_cost[0]))

        #for ele in d_cost:
        #    u_cost += ele
        # u_cost = tf.add_n(d_cost)
        #print('xxxxxxxxx')
        #print("d_cost: " + str(d_cost))
        u_cost = torch.sum(d_cost)
        #print('xxxxxxxxx')
        #print("u_cost: " + str(u_cost.size()))
       
        
        if labeled:
            #print("target: " + str(target.size()))
            #print("h_clean: " + str(h_clean.size()))
            #print("h_corr: " + str(h_corr.size()))
            s_cost = -torch.mean(torch.sum(target*torch.log(h_corr), 1))
            
            #s_cost = torch.nn.functional.cross_entropy(h_corr, target, size_average=False)  # supervised cost
            return u_cost + s_cost

        return u_cost
        
model = VAE()
optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=0.99) #TODO: change to something
#optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum) #TODO: change to something

def target_vector(num): # convert target values into vectors
    t_vec = torch.zeros(num.size()[0], layer_sizes[L])
    for i in range(t_vec.size()[0]):
        val = num[i]
        t_vec[i][val] = 1
    return t_vec
>>>>>>> Stashed changes

#for key in model.a:
#    for param in model.a[key].parameters():
#        print(type(param.data), param.size())

def train(epoch):
    model.train()
<<<<<<< Updated upstream
    train_loss = 0


    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = Variable(data), Variable(target)

        data = data.view(-1,784) #TODO possible reason of bad accuracy

        output,loss = model(data, target,True,False)
        optimizer.zero_grad()
=======
    
    for batch_idx, (data, target) in enumerate(train_loader):
        target = target_vector(target)
        data, target = Variable(data, requires_grad=False), Variable(target, requires_grad=False)
        optimizer.zero_grad()
        loss = model.forward((data, target, True))
>>>>>>> Stashed changes
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
<<<<<<< Updated upstream
        data, target = Variable(data), Variable(target)
        data = data.view(-1, 784)  # TODO possible reason of bad accuracy

        output,loss = model(data, target,False, False )
=======
        target = target_vector(target)    
        data, target = Variable(data, requires_grad=False), Variable(target, requires_grad=False)
>>>>>>> Stashed changes
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if batch_idx % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(unlabeled_train_loader.dataset),
                       100. * batch_idx / len(unlabeled_train_loader), loss.data[0]))


    print("epoch {} completed",epoch)





def test(epoch, valid_loader):

<<<<<<< Updated upstream
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


=======
>>>>>>> Stashed changes
for epoch in range(1, args.epochs + 1):
    train(epoch)
    test(epoch, valid_loader)
