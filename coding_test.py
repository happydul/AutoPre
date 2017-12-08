import torch
from torch.autograd import Variable
from C3D_model import C3D
from Auto_enco import MyAuto
from regression import Regression
import matplotlib.pyplot as plt
from torch import nn, optim
import torch.nn.functional as F
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision import datasets
import time

batch_size = 32
learning_rate = 1e-3
num_epoches = 100

e = torch.rand(6, 8, 8)
b = torch.rand(3, 8, 8)
c = torch.cat([e, b], 0)

'''
#print c.size()
d = c.view(1, 9, 1, 8, 8)
#print d.size()
d = Variable(d)
d = d.cuda()
net = C3D()
net.cuda()
net.eval()
p = net(d)
#print p
#print p.size()
p = p.view(-1, 8, 8)
#print p.size()
'''
# regression test
x = b.view(1, 192)
x = Variable(x)
x.cuda()
print x
with open('./angle.txt', 'r') as f:
    labels = [line.strip() for line in f.readlines()]
length = labels.__len__()
print length

model = Regression(3*8*8, 10, 1)
print model
optimizer = torch.optim.SGD(model.parameters(), lr=0.5)
loss_func = torch.nn.MSELoss()
plt.ion()
prediction = model(x)
print prediction

loss = loss_func(prediction, 0.0)
optimizer.zero_grad()
loss.backward()
optimizer.step()
