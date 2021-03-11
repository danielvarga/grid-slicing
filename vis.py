import matplotlib.pyplot as plt
import numpy as np
import sys

a = np.load(open(sys.argv[1], "rb"))
plt.imshow(a)
plt.show()

print("sum of Lagrangian", a.sum())

n, m = a.shape

def mygrid(shape):
    n, m = shape
    xi = np.arange(n + 1)
    yi = np.arange(m + 1)
    xx, yy = np.meshgrid(xi, yi)
    # g = np.stack([np.arange(n + 1)[:, np.newaxis], np.arange(m + 1)[np.newaxis, :]], axis=-1)
    g = np.stack([xx, yy], axis=-1)
    return g

g = mygrid(a.shape)

x = g[:, :, 0] / n * 2 - 1
y = g[:, :, 1] / m * 2 - 1

dist = np.sqrt(x * x + y * y)

plt.scatter(dist[:-1, :-1].flatten(), a.flatten(), c=np.minimum(np.abs(x[:-1, :-1]), np.abs(y[:-1, :-1])).flatten())
plt.show()


surface = np.exp(-np.abs(np.sqrt(x * x + y * y) - 1)) / 20

plt.imshow(surface)
plt.show()