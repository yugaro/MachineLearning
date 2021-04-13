import matplotlib.pyplot as plt
import torch
import pyro
import pyro.contrib.gp as gp
import pyro.distributions as dist
# import numpy as np
import math
pyro.set_rng_seed(100)

N = 20
X = dist.Uniform(0.0, 5.0).sample(sample_shape=(N,))
y = 0.5 * torch.sin(3 * X) + dist.Normal(0.0, 0.2).sample(sample_shape=(N,))
plt.plot(X.numpy(), y.numpy(), 'kx')
plt.savefig('./image/fig01.pdf')
plt.close()

variance = torch.tensor(1)
lengthscale = torch.tensor([0.1, 0.2, 0.3, 0.5])
noise = torch.tensor(0.01)

kernel = gp.kernels.RBF(input_dim=4, variance=variance,
                        lengthscale=lengthscale)
x1 = torch.tensor([[1, 1, 1, 1]])
x2 = torch.tensor([[1, 1, 1, 1]])
x3 = torch.tensor([[2, 2, 2, 2]])
x4 = torch.tensor([[1, 1, 1, 2]])
x5 = torch.tensor([[1, 1, 2, 2]])
x6 = torch.tensor([[1, 2, 1, 1]])
kernel_metric3 = math.sqrt(kernel(x1, x1) - 2 * kernel(x1, x3) + kernel(x3, x3))
kernel_metric4 = math.sqrt(kernel(x1, x1) - 2 * kernel(x1, x4) + kernel(x4, x4))
kernel_metric5 = math.sqrt(
    kernel(x1, x1) - 2 * kernel(x1, x5) + kernel(x5, x5))
kernel_metric6 = math.sqrt(
    kernel(x1, x1) - 2 * kernel(x1, x6) + kernel(x6, x6))
print(kernel_metric3)
print(kernel_metric5)
print(kernel_metric4)
print(kernel_metric6)

# print(kernel(x1, x2))
# print(kernel(x1, x3))
# print(kernel(x1, x4))

gpr = gp.models.GPRegression(X, y, kernel, noise=noise)
# print(kernel(X, y))
# print(kernel(X, y).size())
# print(X)
# print(y)
# X2 = X.to('cpu').detach().numpy().copy()
# y2 = y.to('cpu').detach().numpy().copy()
# print(np.linalg.norm(X2 - y2, 2))
# print(np.linalg.norm(y2 - X2, 2))

Xtest = torch.linspace(-0.5, 5.5, 500)
with torch.no_grad():
    mean, cov = gpr(Xtest, full_cov=True, noiseless=False)
sd = cov.diag().sqrt()
plt.plot(Xtest.numpy(), mean.numpy(), 'r', lw=2)
plt.fill_between(Xtest.numpy(), (mean - 2.0 * sd).numpy(),
                 (mean + 2.0 * sd).numpy(), color='C0', alpha=0.3)
plt.plot(X.numpy(), y.numpy(), 'kx')
plt.savefig('./image/fig02.pdf')
plt.close()

optimizer = torch.optim.Adam(gpr.parameters(), lr=0.005)
loss_fn = pyro.infer.Trace_ELBO().differentiable_loss
losses = []
num_steps = 2500
for i in range(num_steps):
    optimizer.zero_grad()
    loss = loss_fn(gpr.model, gpr.guide)
    loss.backward()
    optimizer.step()
    losses.append(loss.item())
plt.plot(losses)
plt.xlabel("step")
plt.ylabel("loss")
plt.savefig('./image/fig03.pdf')
plt.close()
# print('variance = {}'.format(gpr.kernel.variance))
# print('lengthscale = {}'.format(gpr.kernel.lengthscale))
# print('noise = {}'.format(gpr.noise))

Xtest = torch.linspace(-0.5, 5.5, 500)
with torch.no_grad():
    mean, cov = gpr(Xtest, full_cov=True, noiseless=False)
sd = cov.diag().sqrt()
plt.plot(Xtest.numpy(), mean.numpy(), 'r', lw=2)
plt.fill_between(Xtest.numpy(), (mean - 2.0 * sd).numpy(),
                 (mean + 2.0 * sd).numpy(), color='C0', alpha=0.3)
plt.plot(X.numpy(), y.numpy(), 'kx')
plt.savefig('./image/fig04.pdf')
plt.close()
