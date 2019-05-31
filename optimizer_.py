import math
import torch
from torch.optim.optimizer import Optimizer
from torch.nn.utils import parameters_to_vector, vector_to_parameters
import torch.nn as nn
import torch.nn.functional as F

def update_input(self, input, output):
    self.input = input[0].data
    self.output = output



class Adam_grad_length(Optimizer):
    

    def __init__(self, model):

            
        self.train_modules = []
        self.set_train_modules(model)
        defaults = dict()
        
        super(Adam_grad_length, self).__init__(model.parameters(), defaults)
        
        for module in self.train_modules:
            module.register_forward_hook(update_input)
            
    def set_train_modules(self, module):
        if len(list(module.children())) == 0:
            if len(list(module.parameters())) != 0:
                self.train_modules.append(module)
        else:
            for child in list(module.children()):
                self.set_train_modules(child)

    def step(self, closure,size):
        
      
        loss = closure()
        linear_combinations = []
        
        # Store the pre-activations
        for module in self.train_modules:
            linear_combinations.append(module.output)

        # Get the gradients w.r.t. pre-activations
        linear_grad = torch.autograd.grad(loss, linear_combinations)
        L = int(len(linear_grad) / 1)
        N = 1
        stacked_linear_grad = []
        for i in range(L):
            lgrad = linear_grad[i]
            for j in range(1, N):
                lgrad = torch.cat([lgrad, linear_grad[i+j*L]])
            stacked_linear_grad.append(lgrad)        
                
        ggn = []
        k=torch.zeros(size).cuda()
        for i, module in enumerate(self.train_modules):
            G = stacked_linear_grad[i]
            A = module.input.clone().detach()
            M = A.shape[0]
            A = torch.cat([A] * N)
            G2 = torch.mul(G, G)

            if isinstance(module, nn.Linear):
                A2 = torch.mul(A, A)
                #k += A2.sum(1).mul(G2.sum(1))
                k+=torch.bmm(G2.unsqueeze(2), A2.unsqueeze(1)).sum((1,2))
                if module.bias is not None:
                    A = torch.ones((M*N, 1), device=A.device)
                    #k+=A.sum(1).mul(G2.sum(1))
                    k+=torch.bmm(G2.unsqueeze(2), A.unsqueeze(1)).sum((1,2))
        
        return k