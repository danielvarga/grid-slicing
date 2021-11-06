import sys
import numpy as np

n, outfilename = sys.argv[1:]
n = int(n)
assert n % 2 == 0
m = n // 2


def line(p1, p2):
    x1, y1 = p1
    x2, y2 = p2
    a = (y2-y1)/(x2-x1)
    # a * x1 + b = y1
    b = y1 - a * x1
    return a, b


def random_line_sparse(n):
    p, q = np.random.uniform(low=0, high=n, size=2) # x_0 y_0 in the paper
    a, b = line((0, p), (m, q))
    cells = []
    for i in range(m):
        j = a * (i + 0.5) + b
        j = int(np.round(j))
        if j < n:
            cells.append((i, j))
    a, b = line((m, q), (n, n-p))
    for i in range(m, n):
        j = a * (i + 0.5) + b
        j = int(np.round(j))
        if j < n:
            cells.append((i, j))
    return cells


def random_line(n):
    s = np.zeros((n, n), dtype=np.uint8)
    p, q = np.random.uniform(low=0, high=n, size=2) # x_0 y_0 in the paper
    a, b = line((0, p), (m, q))
    for i in range(m):
        j = a * (i + 0.5) + b
        j = int(np.round(j))
        if j < n:
            s[i, j] = 1
    a, b = line((m, q), (n, n-p))
    for i in range(m, n):
        j = a * (i + 0.5) + b
        j = int(np.round(j))
        if j < n:
            s[i, j] = 1
    return s


def sampling_collect_lines(n, samples, patience):
    ss = set()
    total_attempts = 0
    attempts = 0
    while True:
        result = random_line(n)
        total_attempts += 1
        attempts += 1
        result_tup = tuple(map(tuple, result))
        before = len(ss)
        ss.add(result_tup)
        if len(ss) > before:
            attempts = 0
        if attempts >= patience:
            break
        if total_attempts >= samples:
            break
        if total_attempts % 1000 == 0:
            print(total_attempts, attempts, len(ss))
    return np.array(list(ss)).astype(np.uint8)


def sampling_collect_lines_sparse(n, samples):
    lines = []
    for i in range(samples):
        lines.append(random_line_sparse(n))
        if i % 1000 == 0:
            print(i)
    return lines


s = random_line(n)
print(s)


do_sparse = True
if do_sparse:
    assert outfilename.endswith("pkl")
    collected_lines = sampling_collect_lines_sparse(n, samples=100000)
    import pickle
    pickle.dump(collected_lines, open(outfilename, "wb"))
else:
    assert outfilename.endswith("npy")
    collected_lines = sampling_collect_lines(n, samples=10000, patience=1000)
    np.save(open(outfilename, "wb"), collected_lines)
