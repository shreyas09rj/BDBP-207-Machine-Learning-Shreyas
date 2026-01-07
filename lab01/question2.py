#!/usr/bin/python

# Implement y = 2x1 + 3 and plot x1, y [start=-100, stop=100, num=100]

import numpy as np

import matplotlib.pyplot as plt
x1 = np.linspace(-100, 100, 100)
y = 2 * x1 + 3


plt.plot(x1, y)
plt.xlabel("x1")
plt.ylabel("y")
plt.title("y = 2x1 + 3")
plt.grid(True)
plt.show()



