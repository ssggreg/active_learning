from Class_vogn import agent_cell
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
    def __init__(self,dropout_rate=None):
        super(type(self), self).__init__()
        self.dropout = dropout_rate
        self.input_size = 9
        
        if dropout_rate:
            self.dropout = nn.Dropout(dropout_rate)
            
        self.layer1 = nn.Linear(9, 64)
        self.layer2 = nn.Linear(64, 64)
        self.layer3 = nn.Linear(64, 1)
        
        self.hidden_layers = nn.ModuleList([self.layer1, self.layer2])
        
        
    def forward(self, x):
        x = F.relu(self.layer1(x))
        if self.dropout:
            x = self.dropout(x)
        x = F.relu(self.layer2(x))
        if self.dropout:
            x = self.dropout(x)
        out = self.layer3(x)
        return out
    
    def forward_gradients(self, x):
        
        x = x.view(-1, self.input_size)
        activation = x
        activations    = [activation]
        preactivations = []
        a = 1
        for layer in self.hidden_layers:
            
            preactivation = layer(activation)
            activation = F.relu(preactivation)
            activations.append(activation)
            
            preactivation.retain_grad()
            preactivation.requires_grad_(True)
            preactivations.append(preactivation)
            
            
        final_preactivation = self.layer3(activation)
        final_preactivation.retain_grad()
        final_preactivation.requires_grad_(True)
        preactivations.append(final_preactivation)
        
       
        return final_preactivation, activations, preactivations
    
    
class EvalNet(nn.Module):

    def __init__(self, in_size=9, hidden=64, dropout_rate=None):
        super(EvalNet, self).__init__()
        self.dropout = dropout_rate
        if dropout_rate:
            self.dropout = nn.Dropout(dropout_rate)
        self.f1 = nn.Linear(in_size, hidden)
        self.f2 = nn.Linear(hidden,  hidden)
        self.f3 = nn.Linear(hidden,  1)

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
#target = ['ACM_G']
X = df[train].values
Y = df[target].values

for i in range(9):
    X[:,i]=(X[:,i] - X[:,i].mean())/(X[:,i].std())
for i in range(len(target)):
    Y[:,i]=(Y[:,i] - Y[:,i].mean())/(Y[:,i].std())
    
trainX_tmp, Xtest, trainY_tmp, Ytest  = train_test_split(X, Y, test_size=0.2, random_state=1)
Xtrain, valX, Ytrain, valY = train_test_split(trainX_tmp, trainY_tmp, test_size=0.25, random_state=1)



'''
__________________________________________________________________________________________________________________-

'''


batch_size_sample = np.ones(261).astype(int)
for i in range(100):
    batch_size_sample[i+110]=3
for i in range(60):
    batch_size_sample[i+200]=5
    
batch_eval = np.ones(261).astype(int)*16

vogn_batch_size =np.ones(261).astype(int)*16

for i in range(100):
    vogn_batch_size[i]=1
for i in range(60):
    vogn_batch_size[i+100]=4
    
nb_ep = np.ones(261).astype(int)*60
for i in range(100):
    nb_ep[i+110]=30
for i in range(60):
    nb_ep[i+200]=30
    
nb_ech = 5
nb_batch=121

loader_batch_size = vogn_batch_size


'''
__________________________________________________________________________________________________________________-

'''


#functions = [reg_std]
#functions = ['grad']
#functions = ['red']
#functions = ['var']
functions = ['max_variance']


places_mc = ['mc_max_variance.npy']

'''
__________________________________________________________________________________________________________________-

'''

if __name__ == '__main__':
    ttt = torch.zeros((5,1000))
    print('vogn')
    mp.set_start_method('spawn')
    for j in range(0):
        processes = []
        print(functions[j])
        for i in range(5):
            torch.manual_seed(i)
            p = mp.Process(target=agent_cell,args=(Xtrain,Ytrain,Xtest,Ytest,BaseNet,EvalNet,functions[j],'VOGN',i,nb_batch,batch_size_sample,nb_ech,vogn_batch_size,batch_eval,nb_ep,ttt))
            p.start()
            processes.append(p)

        for p in processes:
            p.join()
            
        np.save(places_vogn[j], ttt)
        
        
        '''
    __________________________________________________________________________________________________________________-

        '''
        
    ttt = torch.zeros((7,5,1000))
    print('mc')
    for j in range(1):
        for t in range(7):
            print('T',t)
            processes = []
            print(functions[j])
            for i in range(5):
                torch.manual_seed(i)
                p = mp.Process(target=agent_cell,args=(Xtrain,Ytrain[:,t].reshape(-1,1),Xtest,Ytest[:,t].reshape(-1,1),BaseNet,EvalNet,functions[j],'MC',i,nb_batch,batch_size_sample,nb_ech,loader_batch_size,batch_eval,nb_ep,ttt[t]))
                p.start()
                processes.append(p)

            for p in processes:
                p.join()
            
        np.save(places_mc[j], ttt)
        
        
        
        
        
        
        
