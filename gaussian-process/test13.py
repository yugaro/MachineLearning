import numpy as np
import matplotlib.pyplot as plt

def exf(x):
    return 1 - (1 / 2) * ((x ** 2 * np.exp(- x ** 2 / 2)) / (1 - np.exp(- x ** 2 / 2)))

exf_list = []
x = 0.0001
for i in range(100000):
    exfv = exf(x)
    exf_list.append(exfv)
    x += 0.0001

fig, ax = plt.subplots()
ax.plot(exf_list)
plt.show()
