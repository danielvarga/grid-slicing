import matplotlib.pyplot as plt
import numpy as np
import sys
from scipy.ndimage.filters import gaussian_filter


a = np.load(open(sys.argv[1], "rb")).astype(np.float32)

print("symmetrizing")
a = (a + a[:, ::-1]) / 2
a = (a + a[::-1, :]) / 2


plt.imshow(a.T) # , cmap='gist_gray')
plt.show()

print("sum of Lagrangian", a.sum())

n, m = a.shape


b = a[:n // 4, :n // 2]
plt.imshow(b.T) # , cmap='gist_gray')
plt.show()

from mpl_toolkits import mplot3d
fig = plt.figure()
ax = plt.axes(projection='3d')
x = np.linspace(0, 1, n // 2)
y = np.linspace(0, 0.5, n // 4)
X, Y = np.meshgrid(x, y)

b_smoothed = gaussian_filter(b, (2, 2), mode='constant')

# ax.plot_surface(X, Y, b, cmap='viridis', edgecolor='none')
# plt.show()

ax.plot_surface(X, Y, b_smoothed, cmap='viridis', edgecolor='none')
plt.show()
exit()


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


X = np.abs(x.flatten())
Y = np.abs(np.abs(y.flatten()) - 0.5)

# https://stackoverflow.com/a/33966967/383313

# P = np.array([X*0+1, X, Y, X**2, X**2*Y, X**2*Y**2, Y**2, X*Y**2, X*Y]).T

degree = 3

monomials = []
for xd in range(degree + 1):
    for yd in range(degree + 1):
        monomials.append(np.abs(X) ** xd * np.abs(Y) ** yd)
        # monomials.append(X ** xd * Y ** yd)
P = np.array(monomials).T

Q = a.flatten()
print(P.shape, Q.shape)

coeff, r, rank, s = np.linalg.lstsq(P, Q)
print("coeff", coeff, rank, s)

prediction = P.dot(coeff).reshape((n, m))
plt.imshow(prediction.T) # , cmap='gray')
plt.show()

coeff[np.abs(coeff) < 10] = 0
print(coeff)
prediction = P.dot(coeff).reshape((n, m))
plt.imshow(prediction.T)
plt.show()

exit()

dist = np.sqrt(x * x + y * y)

plt.scatter(dist.flatten(), a.flatten(), c=np.minimum(np.abs(x), np.abs(y)).flatten())
plt.title("distance from origin vs value, empirical.\ncolor:distance from axes")
plt.show()

plt.scatter(dist.flatten(), prediction.flatten(), c=np.minimum(np.abs(x), np.abs(y)).flatten())
plt.title("distance from origin vs value, analytical approximation.\ncolor:distance from axes")
plt.show()

l1_dist = np.abs(x) + np.abs(y)

plt.scatter(l1_dist.flatten(), a.flatten(), c=np.minimum(np.abs(x), np.abs(y)).flatten())
plt.title("L1 distance from origin vs value, empirical.\ncolor:distance from axes")
plt.show()