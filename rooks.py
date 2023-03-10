import numpy as np
import sys
from numba import njit

from set_cover import *


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
                else:
                    r[i, j] = 1
            elif c == 1:
                if i > 0:
                    r[i, j] = r[i - 1, j]
                if j > 0:
                    r[i, j] = max((r[i, j], r[i, j - 1]))
            else:
                assert False, "only 0 and 1 allowed"
    return r


def rook_rank(m):
    return dynamic(m)[-1, -1]


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


# without numba: 1e5 calls in 12 seconds.
# with numba: 1e7 calls in 12 seconds.
def speed_test():
    for i in range(100000):
        _ = dynamic(ab)[-1, -1]


def verify_conjecture():
    n = int(sys.argv[1])
    pl = low_slope_gentle_monotones(n) # pl as in positive low slope
    ph = np.transpose(pl, (0, 2, 1)) # ph as in positive high slope
    plph = np.concatenate([pl, ph]) # all positive slope lines
    nh = ph[:, :, ::-1]
    nlnh = np.concatenate([nh, np.transpose(nh, (0, 2, 1))])

    lines = pl

    N = 1000000
    for i in range(1, N+1):
        if i % 10000 == 0:
            print(i, "/", N)
        k = np.random.randint(low=2, high=n - 1)
        selected = np.random.choice(len(lines), size=k, replace=False)
        selected_lines = lines[selected]
        covered = np.amax(selected_lines, axis=0)
        rank = rook_rank(covered)
        if k + rank < n - 1:
            for line in selected_lines:
                print("-----")
                print(pretty(line))
            print("=====")
            print(pretty(covered))
            print(k, "lines +", rank, "rooks <", n, "- 1")
            print(dynamic(covered))

            solve_dual(set_system=nlnh, already_covered=covered)


if __name__ == "__main__":
    verify_conjecture() ; exit()
    # test() ; exit()
    # speed_test() ; exit()
