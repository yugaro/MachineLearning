import numpy as np


def my_kernel(x, x_prime, a, Lambda):
    Lambda_inv = np.linalg.inv(Lambda)
    return 1 * np.exp(- a / 2 * (x - x_prime).T.dot(Lambda_inv).dot(x - x_prime))


def my_kernel_metric(x, x_prime, a, Lambda):
    return np.sqrt(my_kernel(x, x, a, Lambda) - 2 * my_kernel(x, x_prime, a, Lambda) + my_kernel(x_prime, x_prime, a, Lambda))

a = 1
N = 1
eta = 0.1
Lambda = np.diag([0.1, 0.2, 0.3, 0.4])
x1 = np.array([1, 1, 1, 1])
x2 = np.array([2, 1, 1, 1])
x3 = np.array([1, 2, 1, 1])
x4 = np.array([1, 1, 2, 1])


# v = np.array([np.sqrt(1 / 3), np.sqrt(1 / 3), np.sqrt(1 / 3), 0])
# x_q = np.sqrt(2 * np.log(2 * (a ** 2) / (2 * (a ** 2) - ((N * eta) ** 2)))) * np.sqrt(Lambda).dot(v) + x1
# print(x_q)
kernel_metric = my_kernel_metric(x1, x4, a, Lambda)
print(kernel_metric)

# print(x_q)

# x_q_tensor = torch.from_numpy(np.array([x_q]))
# print(x_q_tensor)
# kernel_metric = math.sqrt(
#     kernel(x1, x1) - 2 * kernel(x1, x_q_tensor) + kernel(x_q_tensor, x_q_tensor))
# print(kernel_metric)
