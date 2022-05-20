import numpy as np
import matplotlib.pyplot as plt

def exf(x):
    return 1 - (1 + (np.exp(- x ** 2 / 2) / 2 * (1 - np.exp(- x ** 2 / 2)))) * (x ** 2)

exf_list = []
x = 0.001
x_list = []
for i in range(100000):
    x_list.append(x)
    exfv = exf(x)
    exf_list.append(exfv)
    x += 0.0001
    if exfv > 0:
        print(x, exfv)

fig, ax = plt.subplots()
ax.plot(x_list, exf_list)
plt.show()
