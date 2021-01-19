# Assignment 2 - Rating and Category Prediction; (T3, 2020)
# Completed by Dheeraj Satya Sushant Viswanadham (z5204820)
# Started: 01/11/2020 | Last edited: 15/11/2020
# 
# Note: Used various online resources such as
# Tutorial work, Lecture Notes and Piazza amongst others. I have added the 
# relevant sources above each function where required.
# Good read: https://pytorch.org/tutorials/beginner/blitz/cifar10_tutorial.html
# 
# ------------------------------------------------------------------------------
#
#!/usr/bin/env python3

"""
Question: 
Briefly describe how your program works, and explain any design and training 
decisions you made along the way.

Answer:
I started off with a basic vanilla model with the default parameters. 
To start of with, I decided to go with a simple recurrent neural network (RNN) 
as the order with which the data/sample is processed follows a temporal sequence, 
allowing it to exhibit temporal dynamic behaviour. 

I started off with the tokenisation phase, where I removed any illegal characters 
before processing the text. As part of the pre-processing phase, I converted any 
ratings with numbers (0 - 5) into words to make it more streamlined for processing. 
I also scanned for stopWords within this phase. I did not modify the post-
rocessing function as I felt it was not needed for my model. I kept the word 
vector dimension at 50 initially, before changing it to 200 as I felt it was 
an optimal amount for my model (discussed below).

After processing the sample, I then worked on creating my network. This involved 
creating a special type of RNN called the bidirectional LSTM which is capable of 
learning the long-term dependencies that RNN's aren't capable of overcoming. 
To achieve this, I created two fully connected layers having one that linearises 
to one output and another linearising to 5, depending on whether it's operating 
on either sentiment analysis or multi-class (for the rating and category outputs). 
I then pass my result of my layers into a log_softmax layer and calcuate the loss 
via the loss function from there. This involved experimenting with the dropout 
rates (from 0.5 to 0.0) and I found my optimal value (not mentioned as subject 
to change with further experimentation).

I also decided to go with the Adam optimiser as I found that to be much more 
optimal to converging the loss than SGD. For the parameters, I experimented with 
having different values and found that having trainingValSplit at 0.99 
(which allowed for the model to train on more data), epochs at 6, learning rate 
for the Adam optimiser at 0.001 and having the rest of the parameters at their 
default values, allowed the model to consistently achieve a weighted score of 
85+ (even reaching 87 on one of my runs). 
"""

"""
student.py

UNSW COMP9444 Neural Networks and Deep Learning

You may modify this file however you wish, including creating additional
variables, functions, classes, etc., so long as your code runs with the
hw2main.py file unmodified, and you are only using the approved packages.

You have been given some default values for the variables stopWords,
wordVectors, trainValSplit, batchSize, epochs, and optimiser, as well as a basic
tokenise function.  You are encouraged to modify these to improve the
performance of your model.

The variable device may be used to refer to the CPU/GPU being used by PyTorch.
You may change this variable in the config.py file.

You may only use GloVe 6B word vectors as found in the torchtext package.
"""

import torch
import torch.nn as tnn
import torch.optim as toptim
import numpy as np
import string as st

from itertools import repeat
from torchtext.vocab import GloVe
from config import device


################################################################################
################## The following are my own additional classes #################
################################################################################

# Sources: 
# https://towardsdatascience.com/12-main-dropout-methods-mathematical-and-visual-explanation-58cdc2112293
# and 
# https://www.machinecurve.com/index.php/2019/12/16/what-is-dropout-reduce-overfitting-in-your-neural-networks/#how-well-does-your-model-perform-underfitting-and-overfitting
class SPDT(tnn.Module):
    def __init__(self, rate = 0.2):
        super(SPDT, self).__init__()
        
        self.rate = rate
    
    def forward(self, var, exp_build = None):
        # Duplicate tensor for use with in-place operations (under heavy memory 
        # pressure)
        res = var.clone()
        
        if exp_build is None:
            # Construct the build
            exp_build = (var.shape[0], *repeat(1, var.dim() - 2), var.shape[-1])
        
        self.exp_build = exp_build
        
        if not self.training or self.rate == 0:
            
            return var
            
        else:
            dissonance = self._create_dissonance(var)
            
            if self.rate == 1:
                
                dissonance.fill_(0.0)
                
            else:
                # Bernoulli distribution 
                dissonance.bernoulli_(1 - self.rate).div_(1 - self.rate)
            
            # Two tensors are broadcastable
            dissonance = dissonance.expand_as(var)
            res.mul_(dissonance)
            
            return res
    
    def _create_dissonance(self, var):
        
        return var.new().resize_(self.exp_build)

# Sources: 
# https://towardsdatascience.com/label-smoothing-making-model-robust-to-incorrect-labels-2fae037ffbd0
# and 
# https://pytorch.org/docs/stable/generated/torch.nn.CrossEntropyLoss.html
class LSCE(tnn.Module):
    def __init__(self, rate = 0.1):
        super(LSCE, self).__init__()
        
        self.rate = rate
    
    def forward(self, tensor, out):
        threshold = (1. - self.rate)
        Val_Log = torch.log_softmax(tensor, dim = -1)
        
        expLos = -(Val_Log.gather(dim = -1, index = out.unsqueeze(1)))
        expLos = expLos.squeeze(1)
        
        aveLos = -(Val_Log.mean(dim = -1))
        res = ((threshold * expLos) + (self.rate * aveLos))
        
        return res.mean()


################################################################################
##### The following determines the processing of input data (review text) ######
################################################################################

# Some popular words with no preceived emotional value
stopWords = {
             'this', 'the', 'are', 'item', 'for', 'a', 'an', 'i', "i'm", 'who', 
             'was', 'it', 'and', 'his', 'has', 'to', 'her', 'him', 'his', 'have', 
             'on', 'been', 'is', 'son', 'being', 'thing', 'product', "i'm", 
             'years', 've','first','you', 'of', 'that', "it's", 'they', 'my', 
             'your', 'were'
            }


def tokenise(sample):
    """
    Called before any processing of the text has occurred.
    """
    
    # Remove unnecessary characters from sample by creating a mapping table for 
    # translation
    ill_chars = r"""!"#$%&'()*+:;<=,-./^_`{|}~>?@[\]"""
    res = sample.translate(str.maketrans(ill_chars, " " * len(ill_chars))).split()
    
    return res


def preprocessing(sample):
    """
    Called after tokenising but before numericalising.
    """
    
    processedSample = []
    
    # Convert any numbers present into text format (up to 5)
    TEXTNUM = {'0': "zero", 
               '1': "one", 
               '2': "two", 
               '3': "three", 
               '4': "four", 
               '5': "five"
              }
    
    for elem in sample:
        if elem not in stopWords:
            
            if elem in TEXTNUM.keys():
                processedSample.append(TEXTNUM[elem])
            
            else:
                processedSample.append(elem.strip(st.punctuation))
    
    return processedSample
    
    
def postprocessing(batch, vocab):
    """
    Called after numericalising but before vectorising.
    """
    
    return batch
    
# Adjust the dimension as required
wVDim = 200
wordVectors = GloVe(name = '6B', dim = wVDim)

################################################################################
####### The following determines the processing of label data (ratings) ########
################################################################################

def convertNetOutput(ratingOutput, categoryOutput):
    """
    Your model will be assessed on the predictions it makes, which must be in
    the same format as the dataset ratings and business categories.  The
    predictions must be of type LongTensor, taking the values 0 or 1 for the
    rating, and 0, 1, 2, 3, or 4 for the business category.  If your network
    outputs a different representation convert the output here.
    """
    
    # We need to convert our output to LongTensor by having .long() for 64-bit 
    # integer (signed).Hence, we use argmax for obtaining the index of the 
    # largest value which will convert both tensors to size (batchSize x 1) 
    # as LongTensor [obtained from Piazza].
    ratOut = torch.argmax(input = ratingOutput, dim = 1, keepdim = True).long()
    catOut = torch.argmax(input = categoryOutput, dim = 1, keepdim = True).long()
    
    return ratOut, catOut

################################################################################
###################### The following determines the model ######################
################################################################################

class network(tnn.Module):
    """
    Class for creating the neural network.  The input to your network will be a
    batch of reviews (in word vector form).  As reviews will have different
    numbers of words in them, padding has been added to the end of the reviews
    so we can form a batch of reviews of equal length.  Your forward method
    should return an output for both the rating and the business category.
    """
    
    def __init__(self):
        super(network, self).__init__()
        
        self.spdr = SPDT(rate = 0.2)
        self.rnn = tnn.GRU(input_size = 200, hidden_size = 128, batch_first = True, 
                           bidirectional = True)
        
        # Create layers for rating and business category
        self.full_conn = tnn.Linear(200, 200)
        self.output_rat = tnn.Linear((128 * 2), 2)
        self.output_catHid = tnn.Linear((128 * 2) + 200, 128)
        self.output_cat = tnn.Linear(128, 5)
    
    def forward(self, input, length):
        # Return the mean value of each row of input tensor
        mu_fullConn = torch.mean(input, 1)
        mu_fullConn = torch.relu(self.full_conn(mu_fullConn))
        
        # Get results for rating and business category
        rnn_in = self.rnn(self.spdr(input))[0]
        rnn_res = rnn_in[:, -1, :]
        ratRes = self.output_rat(rnn_res)
        catHid_in = torch.cat([rnn_res, mu_fullConn], dim = 1)
        catHid = torch.relu(self.output_catHid(catHid_in))
        catRes = self.output_cat(catHid)
        
        return ratRes, catRes
        

class loss(tnn.Module):
    """
    Class for creating the loss function.  The labels and outputs from your
    network will be passed to the forward method during training.
    """
    
    def __init__(self):
        super(loss, self).__init__()
        
        self.lf = LSCE(rate = 0)
    
    def forward(self, ratingOutput, categoryOutput, ratingTarget, categoryTarget):
        ratLos = self.lf(ratingOutput, ratingTarget)
        catLos = self.lf(categoryOutput, categoryTarget)
        
        total_L = (ratLos + catLos)
        
        return total_L
        
net = network()
lossFunc = loss()


################################################################################
################## The following determines training options ###################
################################################################################

# My optimal values + Adam optimiser instead of SGD:
trainValSplit = 0.99                                    # Default: 0.8
batchSize = 32                                          # Default: 32
epochs = 6                                              # Default: 10
optimiser = toptim.Adam(net.parameters(), lr = 0.001)   # Default: 
                                                        # toptim.SGD(net.parameters(), 
                                                        #            lr=0.01)

# End of Code