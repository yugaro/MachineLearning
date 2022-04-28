import numpy as np
import matplotlib.pyplot as plt

alpha = 1
Lambda = 1
e = 0
elist = []
dlist = []
for i in range(1000):
    elist.append(e)
    d = np.sqrt(2 * (alpha ** 2) * (1 - np.exp(- 1 / 2 * (e ** 2 / Lambda))))
    dlist.append(d)
    e += 0.01

fig, ax = plt.subplots()
ax.plot(elist, dlist)
ax.scatter(alpha, np.sqrt(2 * (alpha ** 2) * (1 - np.exp(- 1 / 2 * (e ** 2 / Lambda)))), c='r')
plt.show()
