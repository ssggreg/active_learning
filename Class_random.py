from Class_vogn import agent_random
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import torch.nn.functional as F
from torch.utils.data.dataloader import DataLoader
import time
import torch.multiprocessing as mp
from active_function import *
import pandas as pd
from sklearn.model_selection import train_test_split

class BaseNet(nn.Module):
    def __init__(self):
        super(type(self), self).__init__()
        
        self.layer1 = nn.Linear(9, 64)
        self.layer2 = nn.Linear(64, 256)
        self.layer3 = nn.Linear(256, 64)
        self.layer4 = nn.Linear(64, 7)
        
        
    def forward(self, x):
        x = F.relu(self.layer1(x))
        x = F.relu(self.layer2(x))
        x = F.relu(self.layer3(x))
        out = self.layer4(x)
        return out
    
class EvalNet(nn.Module):

    def __init__(self, in_size=9, hidden=64, dropout_rate=None):
        super(EvalNet, self).__init__()
        self.dropout = dropout_rate
        if dropout_rate:
            self.dropout = nn.Dropout(dropout_rate)
        self.f1 = nn.Linear(in_size, hidden)
        self.f2 = nn.Linear(hidden,  hidden)
        self.f3 = nn.Linear(hidden,  7)

    def forward(self, x):
        out = x
        out = F.relu(self.f1(out))
        if self.dropout:
            out = self.dropout(out)
        out = F.relu(self.f2(out))
        if self.dropout:
            out = self.dropout(out)
        out = self.f3(out)
        return out    
    
df = pd.ExcelFile('Test Case 1 - OpAmp_280nm_GF55.xlsx').parse()

train = ['VDD','Ib','Lg1','Lg2','Lrf','Wcf','Wg1','Wg2','temperature']
target = ['ACM_G','SR','CMRR','NOISE','PSRR','PhaseMargin_PM','BandWidth']

X = df[train].values
Y = df[target].values

for i in range(9):
    X[:,i]=(X[:,i] - X[:,i].mean())/(X[:,i].std())
for i in range(7):
    Y[:,i]=(Y[:,i] - Y[:,i].mean())/(Y[:,i].std())
    
#trainX_tmp, Xtest, trainY_tmp, Ytest  = train_test_split(X, Y, test_size=0.2, random_state=1)
#Xtrain, valX, Ytrain, valY = train_test_split(trainX_tmp, trainY_tmp, test_size=0.25, random_state=1)

Xtrain = X
Xtest = X  

Ytrain = Y
Ytest = Y

#for i in range(9):
#    Xtrain[:,i]=(Xtrain[:,i] - Xtrain[:,i].mean())/(Xtrain[:,i].std())
    
#for i in range(9):
#    Xtest[:,i]=(Xtest[:,i] - Xtest[:,i].mean())/(Xtest[:,i].std())

batch_size_sample = np.ones(261).astype(int)
for i in range(100):
    batch_size_sample[i+110]=3
for i in range(60):
    batch_size_sample[i+200]=5
    
vogn_batch_size = np.ones(261).astype(int)*32

batch_eval = np.ones(261).astype(int)*16

for i in range(100):
    vogn_batch_size[i]=1
for i in range(60):
    vogn_batch_size[i+100]=8
    
nb_ep = np.ones(261).astype(int)*50
for i in range(100):
    nb_ep[i+110]=10
for i in range(60):
    nb_ep[i+200]=5
    
nb_ech = 101
nb_batch=261

loader_batch_size = vogn_batch_size 

functions_ = 'random'

places_ ='random.npy'


if __name__ == '__main__':
    ttt = torch.zeros((10,1000))
    print('random')
    mp.set_start_method('spawn')
    for j in range(1):
        processes = []
        print(functions_)
        for i in range(10):
            torch.manual_seed(i)
            p = mp.Process(target=agent_random,args=(Xtrain,Ytrain,Xtest,Ytest,BaseNet,EvalNet,functions_,'lol',i,nb_batch,batch_size_sample,nb_ech,vogn_batch_size,batch_eval,nb_ep,ttt))
            p.start()
            processes.append(p)

        for p in processes:
            p.join()
            
        np.save(places_, ttt)

