# Assignment 1 - Neural Networks and Deep Learning; (T3, 2020)
# Completed by Dheeraj Satya Sushant Viswanadham (z5204820)
# Started: 10/10/2020 | Last edited: 22/10/2020
# 
# Note: Used various online resources such as
# Tutorial work (Exercise 5, Q1) and Lecture Notes.
# 
# ------------------------------------------------------------------------------

# encoder.py
# COMP9444, CSE, UNSW

import torch

star16 = torch.Tensor(
    [[1,1,0,0,0,0,0,0],
     [0,1,1,0,0,0,0,0],
     [0,0,1,1,0,0,0,0],
     [0,0,0,1,1,0,0,0],
     [0,0,0,0,1,1,0,0],
     [0,0,0,0,0,1,1,0],
     [0,0,0,0,0,0,1,1],
     [1,0,0,0,0,0,0,1],
     [1,1,0,0,0,0,0,1],
     [1,1,1,0,0,0,0,0],
     [0,1,1,1,0,0,0,0],
     [0,0,1,1,1,0,0,0],
     [0,0,0,1,1,1,0,0],
     [0,0,0,0,1,1,1,0],
     [0,0,0,0,0,1,1,1],
     [1,0,0,0,0,0,1,1]])

# REPLACE heart18 WITH AN 18-by-14 TENSOR
# TO REPRODUCE IMAGE SHOWN IN SPEC
heart18 = torch.Tensor([
    
    # First create the outer dots on the 4 corners of the square
    [0,0,0,0,   0,0,0,0,    0,0,0,  0,0,0],
    [1,1,1,1,   1,1,1,1,    0,0,0,  0,0,0],
    [0,0,0,0,   0,0,0,0,    1,1,1,  1,1,1],
    [1,1,1,1,   1,1,1,1,    1,1,1,  1,1,1],

    # Form the outer shape of the heart (symmetrical)
    [1,1,0,0,   0,0,0,0,    1,0,0,  0,0,0],
    [1,1,1,0,   0,0,0,0,    1,0,0,  0,0,0],
    [1,1,1,1,   1,0,0,0,    1,0,0,  0,0,0],
    [1,1,1,1,   1,1,0,0,    1,0,0,  0,0,0],
    [1,0,0,0,   0,0,0,0,    1,1,0,  0,0,0],
    [1,1,1,1,   0,0,0,0,    1,1,0,  0,0,0],
    [1,1,1,1,   1,1,1,0,    1,1,0,  0,0,0],
    [1,0,0,0,   0,0,0,0,    1,1,1,  0,0,0],
    [1,1,1,1,   1,1,1,0,    1,1,1,  0,0,0],
    
    # Form the bottom two dots of the heart (symmetrical)
    [1,1,0,0,   0,0,0,0,    1,1,1,  1,0,0],
    [1,1,1,1,   1,1,0,0,    1,1,1,  1,0,0],
    
    # Cool fact: if we only form upto this, we can form a PacMan shape!
    
    # The below two inputs clearly define the outer "curve" shape
    [1,1,1,0,   0,0,0,0,    1,1,1,  1,1,0],
    [1,1,1,1,   1,0,0,0,    1,1,1,  1,1,0],
    
    # Finally, this input creates the bottom "tip" of the heart
    [1,1,1,1,   0,0,0,0,    1,1,1,  1,1,1]])


# REPLACE target1 WITH DATA TO PRODUCE
# A NEW IMAGE OF YOUR OWN CREATION
target1 = torch.Tensor([[0]])

# REPLACE target2 WITH DATA TO PRODUCE
# A NEW IMAGE OF YOUR OWN CREATION
target2 = torch.Tensor([[0]])

# End of Code