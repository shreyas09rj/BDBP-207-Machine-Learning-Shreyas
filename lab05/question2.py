#!/usr/bin/python

# Implement sigmoid function in python and visualize it

# import numpy as np
import matplotlib.pyplot as plt

# def sigmoid(z):
#     return 1 / (1 + np.exp(-z))
# def sigmoid_derivative(z):
#     return sigmoid(z) * (1 - sigmoid(z))
#
# z = np.linspace(-10, 10, 100)
# s = sigmoid(z)
# ds = sigmoid_derivative(z)
# plt.plot(z, s)
# plt.plot(z, ds)
# plt.xlabel('z')
# plt.ylabel('value')
# plt.title('sigmoid function')
# plt.show()



import math
def sigmoid(z):
    return 1 / (1 + math.exp(-z))

def sigmoid_derivat(z):
    return sigmoid(z) * (1 - sigmoid(z))
z = []
for i in range(-100, 101):
    z.append(i /10)

s = []
ds = []

for value in z:
    s.append(sigmoid(value))
    ds.append(sigmoid_derivat(value))

plt.plot(z, s, label="Sigmoid")
plt.plot(z, ds, label="Derivative")
plt.xlabel("z")
plt.ylabel("value")
plt.title("Sigmoid Function")
plt.legend()
plt.show()




