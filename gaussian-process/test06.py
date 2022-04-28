import numpy as np
import matplotlib.pyplot as plt

alpha = 10
b = 2
l = 1000

def f(x):
    return ((alpha * b) ** 2 / l * x * np.exp(-(b * x) ** 2 / (2 * l))) / np.sqrt(2 * alpha ** 2 * (1 - np.exp(- (b * x) ** 2 / (2 * l)))) - 1

xlist = []
x = 100000000000000000000
for i in range(30):
    x = f(x)
    xlist.append(x)

fig, ax = plt.subplots()
ax.plot(xlist, marker='o')
plt.show()
