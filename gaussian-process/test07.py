import numpy as np
import matplotlib.pyplot as plt

alpha = 1
Lambda = 100
b = 1
h = 0.001


def d_func(e):
    return np.sqrt(2 * (alpha ** 2) * (1 - np.exp(- 1 / 2 * ((b * e) ** 2 / Lambda)))) - e


def d_dev(e):
    return (d_func(e + h) - d_func(e)) / h

e = 0
edev_list = []
for i in range(1000):
    edev_list.append(e)
    e += 0.1

fig, ax = plt.subplots()
ax.plot(edev_list)
plt.show()
