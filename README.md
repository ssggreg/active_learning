# active_learning

VOGN:

# Natural-Gradient VI for Bayesian Neural Nets
Minimal code to run Variational Online Gauss-Newton(VOGN) algorithm on MNIST and CIFAR10 datasets using Convolutional Neural Networks. The optimizer is compatible with any PyTorch model containing fully connected or convolutional layers. Layers without trainable parameters such as pooling layers or batch normalization layers with `affine = False` can also be used. 

## Requirements
The project has following dependencies:
- Python >= 3.5
- PyTorch == 1.0
