
import sys
from collections import defaultdict
import numpy as np
from functools import cmp_to_key

import matplotlib.pyplot as plt

sizes = []
for n in range(2, 28):
    a = np.load("%d-%d.npy" % (n, n))
    print(a.dtype) ; exit()
    sizes.append((n, len(a)))

sizes = np.array(sizes)

x = sizes[:, 0]
y = sizes[:, 1]
plt.plot(x, y)
# plt.plot(x, x ** 4 * 0.37)
# plt.plot(x, x ** 4 * 0.37 / np.log(x) * 3)

P = np.array([x*0 + 1, x, x ** 2, x ** 3, x ** 4]).T

coeff, r, rank, s = np.linalg.lstsq(P, y)
print(coeff, rank, s)
prediction = P.dot(coeff)
plt.plot(x, prediction)

plt.plot(x, 1.31119651 * x ** 3 + 0.30228286 * x ** 4)

plt.show()
