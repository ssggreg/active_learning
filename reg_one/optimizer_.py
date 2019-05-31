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
        L = len(linear_grad)                      
        ggn = []
        
        k=torch.zeros(size).cuda()
        
        for i, module in enumerate(self.train_modules):
            G = linear_grad[i]
            A = module.input.clone().detach()
            M = A.shape[0]
            G2 = torch.mul(G, G)
            A2 = torch.mul(A, A)
            

            Ra = A2.sum(1)
            Rh = G2.sum(1)
            #k+=torch.bmm(G2.unsqueeze(2), A2.unsqueeze(1)).sum((1,2))
            k+=Rh.mul(Ra)
            if module.bias is not None:

                A = torch.ones((M, 1), device=A.device)
                Ra = A.sum(1)
                k+=Rh.mul(Ra)
                #k+=torch.bmm(G2.unsqueeze(2), A.unsqueeze(1)).sum((1,2))

        return k
    
    
    
    