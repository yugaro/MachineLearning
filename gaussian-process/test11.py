import numpy as np
import matplotlib.pyplot as plt

def f(x):
    return x / np.sqrt(1 / (np.exp(2 * 1000 * (x ** 2)) - np.exp(- 1000 * x ** 2)))

x_list = []
y_list = []
x = 0
for i in range(100):
    x_list.append(x)
    y = f(x)
    y_list.append(y)
    x += 0.001

fig, ax = plt.subplots()
ax.plot(x_list, y_list)
plt.show()
