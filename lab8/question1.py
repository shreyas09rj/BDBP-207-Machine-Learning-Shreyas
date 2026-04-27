#!/usr/bin/python
# Regularization
# From scrath


import numpy as np

def l1_norm(w):
    return np.sum(np.abs(w))

def l2_norm(w):
    return np.sqrt(np.sum(w**2))

w = np.array([1, -2, 3])

print("L1 Norm:", l1_norm(w))
print("L2 Norm:", l2_norm(w))
