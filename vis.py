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
    xi = np.arange(n)
    yi = np.arange(m)
    xx, yy = np.meshgrid(xi, yi)
    # g = np.stack([np.arange(n + 1)[:, np.newaxis], np.arange(m + 1)[np.newaxis, :]], axis=-1)
    g = np.stack([xx, yy], axis=-1)
    return g

g = mygrid(a.shape)

x = g[:, :, 0] / (n - 1) * 2 - 1
y = g[:, :, 1] / (m - 1) * 2 - 1


X = x.flatten()
Y = y.flatten()

P = np.array([X*0+1, X, Y, X**2, X**2*Y, X**2*Y**2, Y**2, X*Y**2, X*Y]).T
Q = a.flatten()
print(P.shape, Q.shape)

coeff, r, rank, s = np.linalg.lstsq(P, Q)
print(coeff, rank, s)

prediction = P.dot(coeff).reshape((n, m))
plt.imshow(prediction)
plt.show()


coeff[np.abs(coeff) < 1e-3] = 0
print(coeff)
prediction = P.dot(coeff).reshape((n, m))
plt.imshow(prediction)
plt.show()


dist = np.sqrt(x * x + y * y)

plt.scatter(dist.flatten(), a.flatten(), c=np.minimum(np.abs(x), np.abs(y)).flatten())
plt.title("distance from origin vs value, empirical.\ncolor:distance from axes")
plt.show()

plt.scatter(dist.flatten(), prediction.flatten(), c=np.minimum(np.abs(x), np.abs(y)).flatten())
plt.title("distance from origin vs value, analytical approximation.\ncolor:distance from axes")
plt.show()
