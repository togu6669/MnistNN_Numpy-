# Examples : 
# https://nextjournal.com/gkoehler/pytorch-mnist
# https://towardsdatascience.com/handwritten-digit-mnist-pytorch-977b5338e627

# fast MNIST https://gist.github.com/y0ast/f69966e308e549f013a92dc66debeeb4

import os
import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from torch.autograd import Variable
import torchvision.datasets as dset
import torchvision.transforms as transforms
from timeit import default_timer as timer

# ...

class FastMNIST(dset.MNIST):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
        # Scale data to [0,1]
        self.data = self.data.unsqueeze(1).float().div(255)
        
        # Normalize it with the usual MNIST mean and std
        self.data = self.data.sub_(0.1307).div_(0.3081)
        
        # Put both data and targets on GPU in advance
        device = torch.device('cuda')
        self.data, self.targets = self.data.to(device), self.targets.to(device)

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        img, target = self.data[index], self.targets[index]

        return img, target

use_cuda = torch.cuda.is_available()

# load mnist dataset
root = './data'
if not os.path.exists(root):
    os.mkdir(root)

# test if torch is using GPU / CUDA
# print(torch.rand(2,3).cuda())
# torch.cuda.is_available()
# use_cuda = False

# if use_cuda:
#     print(torch.cuda.get_device_name(0))
#     print('Memory Usage:')
#     print('Allocated:', round(torch.cuda.memory_allocated(0)/1024**3,1), 'GB')
#     print('Cached:   ', round(torch.cuda.memory_cached(0)/1024**3,1), 'GB')
#     print('Summary:   ', torch.cuda.memory_summary(0))
    
def fastLoader ():

    train_dataset = FastMNIST(root, train=True, download=True)
    test_dataset = FastMNIST(root, train=False, download=True)

    train_l = torch.utils.data.DataLoader(train_dataset, batch_size=64, shuffle=True, num_workers=0)
    test_l = torch.utils.data.DataLoader(test_dataset, batch_size=10000, shuffle=False, num_workers=0)
    return train_l, test_l


def standardLoader():

    trans = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (1.0,))])
    # if not exist, download mnist dataset
    train_set = dset.MNIST(root=root, train=True, transform=trans, download=True)
    test_set = dset.MNIST(root=root, train=False, transform=trans, download=True)

    batch_size = 100

    train_l = torch.utils.data.DataLoader(
                    dataset=train_set,
                    batch_size=batch_size,
                    shuffle=True)
    test_l = torch.utils.data.DataLoader(
                    dataset=test_set,
                    batch_size=batch_size,
                    shuffle=False)
    return train_l, test_l

# train_loader, test_loader = standardLoader()
train_loader, test_loader = fastLoader()

print('==>>> total trainning batch number: {}'.format(len(train_loader)))
print('==>>> total testing batch number: {}'.format(len(test_loader)))

## class LeNet(nn.Module):
#     def __init__(self):
#         super(LeNet, self).__init__()
#         self.conv1 = nn.Conv2d(1, 20, 5, 1)
#         self.conv2 = nn.Conv2d(20, 50, 5, 1)
#         self.fc1 = nn.Linear(4*4*50, 500)
#         self.fc2 = nn.Linear(500, 10)

#     def forward(self, x):
#         x = F.relu(self.conv1(x))
#         x = F.max_pool2d(x, 2, 2)
#         x = F.relu(self.conv2(x))
#         x = F.max_pool2d(x, 2, 2)
#         x = x.view(-1, 4*4*50)
#         x = F.relu(self.fc1(x))
#         x = self.fc2(x)
#         return x
    
#     def name(self):
#         return "LeNet"

# class MLPNet(nn.Module):
#     def __init__(self):
#         super(MLPNet, self).__init__()
#         self.fc1 = nn.Linear(28*28, 500)
#         self.fc2 = nn.Linear(500, 256)
#         self.fc3 = nn.Linear(256, 10)
#     def forward(self, x):
#         x = x.view(-1, 28*28)
#         x = F.relu(self.fc1(x))
#         x = F.relu(self.fc2(x))
#         x = self.fc3(x)
#         return x
    
#     def name(self):
#         return "MLP"

class MLPNet(nn.Module):
    def __init__(self):
        super(MLPNet, self).__init__()
        self.fc1 = nn.Linear(28*28, 128)
        self.fc2 = nn.Linear(128, 10)
    
    def forward(self, x):
        x = x.view(-1, 28*28)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x
    
    def name(self):
        return "MLP"



# other network initialization in torch

# input_size = 784
# hidden_sizes = [128, 64]
# output_size = 10

# model = nn.Sequential(nn.Linear(input_size, hidden_sizes[0]),
#                       nn.ReLU(),
#                       nn.Linear(hidden_sizes[0], hidden_sizes[1]),
#                       nn.ReLU(),
#                       nn.Linear(hidden_sizes[1], output_size),
#                       nn.LogSoftmax(dim=1))
# print(model)


start = timer()

# training
model = MLPNet()

if use_cuda:
    model = model.cuda()

optimizer = optim.Adam(model.parameters(), lr=0.001)
# optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)

criterion = nn.CrossEntropyLoss()

for epoch in range(5):
    # trainning
    ave_loss = 0
    for batch_idx, (x, target) in enumerate(train_loader):
        optimizer.zero_grad()
        if use_cuda:
            x, target = x.cuda(), target.cuda()
        x, target = Variable(x), Variable(target)
        out = model(x)
        loss = criterion(out, target)
        ave_loss = ave_loss * 0.9 + loss.data.item() * 0.1
        loss.backward()
        optimizer.step()
        if (batch_idx+1) % 100 == 0 or (batch_idx+1) == len(train_loader):
            print('==>>> epoch: {}, batch index: {}, train loss: {:.6f}'.format(
                epoch, batch_idx+1, ave_loss))
    # testing
    correct_cnt, ave_loss = 0, 0
    total_cnt = 0
    for batch_idx, (x, target) in enumerate(test_loader):
        if use_cuda:
            x, target = x.cuda(), target.cuda()
        x, target = Variable(x, volatile=True), Variable(target, volatile=True)
        out = model(x)
        loss = criterion(out, target)
        _, pred_label = torch.max(out.data, 1)
        total_cnt += x.data.size()[0]
        correct_cnt += (pred_label == target.data).sum()
        # smooth average
        ave_loss = ave_loss * 0.9 + loss.data.item() * 0.1
        
        if(batch_idx+1) % 100 == 0 or (batch_idx+1) == len(test_loader):
            print( '==>>> epoch: {}, batch index: {}, test loss: {:.6f}, acc: {:.3f}'.format(
                epoch, batch_idx+1, ave_loss, correct_cnt * 1.0 / total_cnt))

# torch.save(model.state_dict(), model.name())
end = timer()
sectime = end - start
mintime = sectime / 60
print(' Time of training : ', sectime, ' sec, ', mintime, ' min')
