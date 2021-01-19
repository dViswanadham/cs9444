# Assignment 1 - Neural Networks and Deep Learning; (T3, 2020)
# Completed by Dheeraj Satya Sushant Viswanadham (z5204820)
# Started: 10/10/2020 | Last edited: 22/10/2020
# 
# Note: Used various online resources such as https://github.com/rois-codh/kmnist#kuzushiji-mnist-1
# for information about the Kuzushiji-MNIST dataset, as well as
# https://pytorch.org/docs/stable/generated/torch.nn.Linear.html
# for pytorch documentation and Tutorial work and Lecture Notes.
# 
# ------------------------------------------------------------------------------

# kuzu.py
# COMP9444, CSE, UNSW

from __future__ import print_function
import torch
import torch.nn as nn
import torch.nn.functional as F

class NetLin(nn.Module):
    # linear function followed by log_softmax
    def __init__(self):
        super(NetLin, self).__init__()
        
        # INSERT CODE HERE
        # Structure of NN:
        # 28 x 28 x 1 = 784 size grayscale images
        self.lay1 = nn.Linear(in_features = 784, out_features = 10)
        

    def forward(self, x):
        # First flatten the input for processing
        flat = x.view(x.shape[0], -1)
        
        fin = F.log_softmax(self.lay1(flat), dim = 1)
        
        return fin

class NetFull(nn.Module):
    # two fully connected tanh layers followed by log softmax
    def __init__(self):
        super(NetFull, self).__init__()
        
        # INSERT CODE HERE
        # Trialling different number of hidden nodes (multiples of 10)
        self.lay1 = nn.Linear(in_features = 784, out_features = 190)
        self.lay2 = nn.Linear(in_features = 190, out_features = 10)
        
        
    def forward(self, x):
        flat = x.view(x.shape[0], -1)
        tanflat = torch.tanh(self.lay1(flat))
        
        fin = F.log_softmax(self.lay2(tanflat), dim = 1)
        
        return fin

class NetConv(nn.Module):
    # two convolutional layers and one fully connected layer,
    # all using relu, followed by log_softmax
    def __init__(self):
        super(NetConv, self).__init__()
        
        # INSERT CODE HERE
        # Two convolution layers:                       # (Outdated) Old Workings:
        self.cv1 = nn.Conv2d(1, 64, 5, padding = 2)     # 28 x 28 x 1 -> 28 - 5 + (2 x 2) + 1 = 28
        self.cv2 = nn.Conv2d(64, 32, 5, padding = 2)    # 28 x 28 x 16 -> (28+2)/5 = 6 x 6 x 16
                                                        # 6 - 5 + (2 x 2) + 1 = 6 x 6 x 32
                                                        # (6+2)/5 = 1.6 x 1.6 x 32
        # Fully connected layer
        self.lay1 = nn.Linear(in_features = 1152, out_features = 650)
        self.lay2 = nn.Linear(in_features = 650, out_features = 10)
        
        self.maxpl = nn.MaxPool2d(5, padding = 2)
        
        
    def forward(self, x):
        relact = self.maxpl(F.relu(self.cv2(F.relu(self.cv1(x)))))
        flat = relact.view(relact.shape[0], -1)
        fin = F.log_softmax(self.lay2(F.relu(self.lay1(flat))), dim = 1)
        
        return fin

# End of Code