import matplotlib.pyplot as plt
import numpy as np
import sys

cmap = "gist_gray"
cmap = "Blues"


def mu1(x, y):
    return 3/4 * (x ** 2 - 2 * x **2 * y ** 2 + y ** 2)

def mu2(x, y):
    x = np.abs(x)
    y = np.abs(y)
    return 0.3 * (x + y) + 0.43 * (x ** 3 + y ** 3) - 0.585 * (x ** 3 * y + y ** 3 * x) - 0.16 * x ** 2 * y ** 2


x = np.linspace(-1, 1, 100)
y = np.linspace(-1, 1, 100)
X, Y = np.meshgrid(x, y)
Z = mu1(X, Y)
fig, ax = plt.subplots(figsize=(6, 6))
ax.imshow(Z, cmap=cmap, extent=[-1, 1, -1, 1])
plt.xticks([-1, -0.5, 0, 0.5, 1])
plt.yticks([-1, -0.5, 0, 0.5, 1])
ax.set_aspect("equal")
plt.subplots_adjust(left=0.08, right=0.92, bottom=0.08, top=0.92)
plt.savefig("mu1.pdf")
plt.show()


Z = mu2(X, Y)
fig, ax = plt.subplots(figsize=(6, 6))
ax.imshow(Z, cmap=cmap, extent=[-1, 1, -1, 1])
plt.xticks([-1, -0.5, 0, 0.5, 1])
plt.yticks([-1, -0.5, 0, 0.5, 1])
ax.set_aspect("equal")
plt.subplots_adjust(left=0.08, right=0.92, bottom=0.08, top=0.92)
plt.savefig("mu2.pdf")
plt.show()


a = np.load("dual-tensorflow.300-300.npy")
print("symmetrizing")
a = (a + a[:, ::-1]) / 2
a = (a + a[::-1, :]) / 2

fig, ax = plt.subplots(figsize=(6, 6))
plt.imshow(a.T, cmap=cmap)
plt.subplots_adjust(left=0.08, right=0.92, bottom=0.08, top=0.92)
plt.xticks(range(0, 301, 50))
plt.yticks(range(0, 301, 50))
plt.savefig("dualweight.pdf")
plt.show()


a = np.load("lagrangian-tensorflow.30-30.middle.npy")
print("symmetrizing")
a = (a + a[:, ::-1]) / 2
a = (a + a[::-1, :]) / 2
a = (a + a.T) / 2

fig, ax = plt.subplots(figsize=(6, 6))
plt.imshow(a, cmap=cmap)
plt.subplots_adjust(left=0.08, right=0.92, bottom=0.08, top=0.92)
plt.xticks(range(0, 31, 5))
plt.yticks(range(0, 31, 5))
plt.savefig("weight_30_A.pdf", dpi=600)
plt.show()
