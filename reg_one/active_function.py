import numpy as np
from scipy import stats

def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def f (x):
    x = (x>0).astype(int)
    x = np.mean(x,axis = 0)
    return x*(1-x)

def max_entropy (x):
    x = sigmoid(x)
    p = np.mean(x,axis = 0)
    pp = np.ones(p.shape)-p
    return -p*np.log(p+1e-10)-pp*np.log(pp+1e-10)

def bald (x) :     
    sx = sigmoid(x)
    return max_entropy (x) - np.mean(sx * np.log(sx + 1e-10) + (1 - sx) * np.log(1 - sx + 1e-10), axis=0)    
    
def var_ratios(x):
    x.shape[1]
    preds = (x >= 0).astype(int)
    mode, count = stats.mode(preds, axis=0)
    return  (1 - count / x.shape[0]).reshape((-1,))
                
def mean_std(x):
    x = sigmoid(x)
    return x.std(axis=0)

def random(x):
    return np.random.rand(x.shape[-1])

def reg_std(x):
    return np.mean(np.std(x,axis=0),axis=0)

def emc_pre(dataloader,criterion,size,model,optimizer):
    L = np.zeros((size))
    a=0
    for i in dataloader:
        inputs = i['data']
        labels = i['label']
        inputs, labels = inputs.cuda(), labels.cuda()
        optimizer.zero_grad()
        logits = model.forward(inputs)
        L = criterion(logits, labels)
    
    return L


def expected_gradient_length(X_pool,model):
    x_pool  = torch.tensor(X_pool, dtype=torch.float).cuda()
   
    F, activations, preactivations = model.forward_gradients(x_pool)
    
    F = F.view(x_pool.shape[0],7,1).expand(x_pool.shape[0],7,100)
    
    Fhat = torch.stack([active.bnn(x_pool) for _ in range(100)], dim=-1)
    
    Cost = (1. / 100) * torch.pow(F - Fhat, 2).sum().sum().sum()
    
    preactivations_grads = torch.autograd.grad(Cost, preactivations)
    gradients = goodfellow_backprop(activations, preactivations_grads)
    squared_L2 = torch.zeros(x_pool.shape[0], requires_grad=False)
    
    for gradient in gradients:
        Sij = torch.pow(gradient, 2).sum(dim=tuple(range(1,gradient.dim())))
        squared_L2 += Sij.cpu()
        
    return squared_L2



def goodfellow_backprop(activations, linearGrads):
    grads = []
    for i in range(len(linearGrads)):
        G, X = linearGrads[i], activations[i]
        if len(G.shape) < 2:
            G = G.unsqueeze(1)

        G *= G.shape[0] # if the function is an average

        grads.append(torch.bmm(G.unsqueeze(2), X.unsqueeze(1)))
        grads.append(G)

    return grads



