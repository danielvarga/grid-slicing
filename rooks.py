import numpy as np
import sys
from numba import njit


@njit
def dynamic(m):
    r = np.zeros_like(m)
    n, n_ = m.shape
    assert n == n_
    for i in range(n):
        for j in range(n):
            c = m[i, j]
            if c == 0:
                if i > 0 and j > 0:
                    r[i, j] = r[i - 1, j - 1] + 1
            elif c == 1:
                if i > 0:
                    r[i, j] = r[i - 1, j]
                if j > 0:
                    r[i, j] = max((r[i, j], r[i, j - 1]))
            else:
                assert False, "only 0 and 1 allowed"
    return r


def pretty(s):
    return str(s).replace('1', '■').replace('0', '·').replace('[', ' ').replace(']', ' ').replace(' ', '')


def test():
    ss = np.load("output/solution-mono.7.53850.npy").astype(np.int32)
    print(ss.shape)
    ab = 1 - (1-ss[0]) * (1-ss[1])
    cde = 1 - (1-ss[2]) * (1-ss[3]) * (1-ss[4])
    cde = cde[:, ::-1] # we want the placed lines to ascend (that's descend on the pretty() visualization, sorry!

    print(pretty(ab))
    print(dynamic(ab))

    print("====")
    print(pretty(cde))
    print(dynamic(cde))


# with numba: 1e7 calls in 12 seconds.
# without numba: 1e5 calls in 12 seconds.
# for i in range(100000): _ = dynamic(ab)[-1, -1]

test() ; exit()
