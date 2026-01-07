#!/usr/bin/python
# Implement Gaussian PDF - mean = 0, sigma = 15 in the range[start=-100, stop=100, num=100]

import numpy as np
import matplotlib.pyplot as plt

mu = 0
sigma = 15

x = np.linspace(-100, 100, 100)
pdf = (1 / (sigma * np.sqrt(2 * np.pi))) * np.exp(-((x - mu) ** 2) / (2 * sigma ** 2))

plt.plot(x, pdf)
plt.xlabel("x")
plt.ylabel("Gaussian PDF")
plt.grid(True)
plt.show()




