import numpy as np
import matplotlib.pyplot as plt

alpha = 22
Lambda = 100000
b = 2


def d_func(e):
    return np.sqrt(2 * (alpha ** 2) * (1 - np.exp(- 1 / 2 * ((b * e)**2 / Lambda))))

e = 10000000000
e_list = []
for i in range(100):
    e_list.append(e)
    e = d_func(e)

fig, ax = plt.subplots()
ax.plot(e_list)
plt.show()
