import numpy as np
import matplotlib.pyplot as plt

alpha = 10
b = 10
Lambda = 10000
delta = 0.00000001

def kernel(e):
    return alpha ** 2 * np.exp(- (b * e) ** 2 / (2 * Lambda))


def km(e):
    return np.sqrt(2 * (alpha ** 2) - 2 * kernel(e))

def dh(e):
    return (kernel(e) / km(e)) * (b ** 2 / Lambda) * e - 1


def dev(e):
    return (dh(e + delta) - dh(e)) / delta

def ddh(e):
    return 1 - (1 + (kernel(e) / (km(e) ** 2))) * (b ** 2 / Lambda) * e

# e = 10
# e_list = [e]
# for i in range(1000):
#     e_next = km(e)
#     e_list.append(e_next)
#     e = e_next
# fig, ax = plt.subplots()
# ax.plot(e_list)
# plt.show()

e = 0.001
e_list = []
ev_list = []
for i in range(100000):
    e_list.append(e)
    ev = dh(e)
    ev_list.append(ev)
    e += 0.001
fig, ax = plt.subplots()
ax.plot(e_list, ev_list)
plt.show()

# e = 0.001
# e_list = []
# ddh_list = []
# for i in range(1000):
#     e_list.append(e)
#     ddhv = dev(e)
#     ddh_list.append(ddhv)
#     e += 0.001
# fig, ax = plt.subplots()
# ax.plot(e_list, ddh_list)
# plt.show()

# e = 0.001
# e_list = []
# ddh_list = []
# for i in range(1000):
#     e_list.append(e)
#     ddhv = ddh(e)
#     ddh_list.append(ddhv)
#     e += 0.001
# fig, ax = plt.subplots()
# ax.plot(e_list, ddh_list)
# plt.show()
