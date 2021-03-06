import sys
from collections import defaultdict
import numpy as np
from functools import cmp_to_key

import matplotlib.pyplot as plt

import SetCoverPy.setcover as setcover


def sort_fractions(a):
    def comparator(x, y):
        a, b = x # a/b
        c, d = y # c/d
        return a * d - c * b
    return sorted(a, key=cmp_to_key(comparator))

def test_sort_fractions():
    print(sort_fractions([(3, 2), (4, 3), (-1, 2)]))


def tof(x): # to float
    return x[0] / x[1]


# not normalized!
def average_fractions(x, y):
    a, b = x[..., 0], x[..., 1] # a/b
    c, d = y[..., 0], y[..., 1] # c/d
    # a/b + c/d = (ad + cb) / 2bd
    return np.stack([a * d + c * b, 2 * b * d], axis=-1)

def test_average_fractions():
    for _ in range(1000):
        a = np.random.randint(low=-10, high=10, size=(2, 2))
        if a[0, 1] == 0 or a[1, 1] == 0:
            continue
        x = normalize(a[0, :])
        y = normalize(a[1, :])
        avg = average_fractions(x, y)
        assert np.isclose(tof(avg), (tof(x) + tof(y)) / 2)


def mygrid(shape):
    n, m = shape
    xi = np.arange(n + 1)
    yi = np.arange(m + 1)
    xx, yy = np.meshgrid(xi, yi)
    # g = np.stack([np.arange(n + 1)[:, np.newaxis], np.arange(m + 1)[np.newaxis, :]], axis=-1)
    g = np.stack([xx, yy], axis=-1)
    return g


# todo vectorize
def normalize(frac):
    a, b = frac
    assert b != 0
    if b < 0:
        a, b = -a, -b
    g = np.gcd(a, b)
    return a // g, b // g


def pretty(s):
    return str(s.astype(int)).replace('1', '■').replace('0', '·').replace('[', ' ').replace(']', ' ').replace(' ', '')


def parametrized_monotone(shape, jumps):
    n, m = shape
    assert len(jumps) == n + 1
    assert min(jumps) >= 0
    assert max(jumps) <= m
    result = np.zeros((n, m), dtype=np.uint8)
    y_prev = None
    for x, y in enumerate(jumps): # x addresses a row, y a column!
        if x > 0:
            result[x - 1, y_prev : y + 1] = 1
        y_prev = y
    return result

n = 5
shape = n, n

# jumps = [0, 0, 1, 2, 3, 4] ; print(pretty(parametrized_monotone(shape, jumps))) ; exit()

def sample_monotones(shape, sample_count):
    n, m = shape
    results = []
    hashes = set()
    results = []
    for i in range(sample_count):
        jumps = np.sort(np.random.choice(m, n + 1))
        result = parametrized_monotone(shape, jumps)
        result_tup = tuple(map(tuple, result))
        h = hash(result_tup)
        if h not in hashes:
            hashes.add(h)
            results.append(result)
        if i % 10000 == 0:
            print(i, len(hashes))
    '''
    for result in results:
        print("====")
        print(pretty(result))
    '''
    results = np.array(results)
    return results

# sample_monotones(shape, sample_count=100000) ; exit()


def slow_slices(line, shape):
    n, m = shape
    a, b, c = line
    ev = a * np.arange(n + 1)[:, np.newaxis] + b * np.arange(m + 1)[np.newaxis, :] + c
    result = np.zeros((n, m), dtype=np.uint8)
    for x in range(n):
        for y in range(m):
            evs = [ev[x, y], ev[x+1, y], ev[x, y+1], ev[x+1, y+1]]
            if min(evs) < 0 and max(evs) > 0:
                result[x, y] = 1
    return result


def slices(line, shape):
    n, m = shape
    a, b, c = line
    ev = a * np.arange(n + 1)[:, np.newaxis] + b * np.arange(m + 1)[np.newaxis, :] + c
    evs = np.stack([ev[:-1, :-1], ev[1:, :-1], ev[:-1, 1:], ev[1:, 1:]], axis=-1)
    lows = np.min(evs, axis=2) < 0
    highs = np.max(evs, axis=2) > 0
    result = np.logical_and(lows, highs)
    return result


def test_fast_slicing_implementation():
    shape = (10, 10)
    for i in range(10000):
        a, b = np.random.normal(size=2)
        c = np.random.normal(size=1)
        line = a, b, c
        result = slices(line, shape)
        slow_result = slow_slices(line, shape)
        assert np.all(result == slow_result)

# test_fast_slicing_implementation() ; exit()


# returns all rational numbers x in [0, n] for which
# there are two points on the grid collinear with (x, 0).
#
# this broadcasting-based algorithm is very fast
# but goes to hell memory-wise for large ns.
# don't try with n > 100.
def intersect_all_with_side(shape):
    g = mygrid(shape)
    n, m = shape
    nc, mc = n + 1, m + 1
    x1 = g[:, :, 0].reshape((nc, mc, 1, 1))
    y1 = g[:, :, 1].reshape((nc, mc, 1, 1))
    x2 = g[:, :, 0].reshape((1, 1, nc, mc))
    y2 = g[:, :, 1].reshape((1, 1, nc, mc))

    print("broadcasting")
    dividents = x1 * y2 - x2 * y1
    divisors = y2 - y1
    print("broadcasted")

    nonzero = divisors != 0
    dividents = dividents[nonzero]
    divisors = divisors[nonzero]
    print("zero-filtered")

    g = np.gcd(dividents, divisors)
    dividents //= g
    divisors //= g
    print("simplified")

    signs = np.sign(divisors)
    dividents *= signs
    divisors *= signs
    print("sign-simplified")

    nonnegative = dividents >= 0
    dividents = dividents[nonnegative]
    divisors = divisors[nonnegative]
    print("nonnegative-filtered")

    bounded = dividents <= n * divisors
    dividents = dividents[bounded]
    divisors = divisors[bounded]
    print("bounded-filtered")

    fractions = np.stack([dividents, divisors], axis=-1)
    print(fractions.shape)
    fractions = np.unique(fractions, axis=0)
    print("unique")
    sorted_fractions = np.array(sort_fractions(fractions))
    print("sorted")
    return sorted_fractions


# todo normalize, but i guess it's never in fact unnormalized
def region_centers(cutpoints):
    return average_fractions(cutpoints[:-1], cutpoints[1:])


def cross(p1, p2):
    return p1[0] * p2[1] - p2[0] * p1[1]

def sweep(shape, fraction):
    # unused, just for reference
    f = tof(fraction)
    def floating_point_comparator(p1, p2):
        p1tr = p1[0] - f, p1[1]
        p2tr = p2[0] - f, p2[1]
        return np.sign(cross(p1tr, p2tr))

    p, q = fraction
    def comparator(p1, p2):
        # ts as in translated and scaled
        p1ts = p1[0] * q - p, p1[1] * q
        p2ts = p2[0] * q - p, p2[1] * q
        return np.sign(cross(p1ts, p2ts))

    n, m = shape
    g = mygrid((n, m)).reshape((-1, 2))

    ordered = sorted(g, key=cmp_to_key(comparator))
    return ordered


def test_sweep():
    shape = 50, 51
    fraction = 51, 2
    ordered = np.array(sweep(shape, fraction))
    plt.scatter(ordered[:, 0], ordered[:, 1], c=range(len(ordered)))
    plt.show()

# test_sweep() ; exit()

def line_through(fraction, point):
    p, q = fraction # interpreted as point (p/q, 0)
    x, y = point
    # solving
    # 1. a*x + b*y + c = 0
    # 2. a*p/q + b*0 + c = 0
    # in integers (a, b, c) yields this:
    a = - q * y
    c = p * y
    b = - p + q * x
    return a, b, c


# line connecting (fraction, 0) and (point1+point2)/2
def line_between(fraction, point1, point2):
    p, q = fraction # interpreted as point (p/q, 0)
    X, Y = (point1[0] + point2[0], point1[1] + point2[1])
    # solving
    # 1. a*X/2 + b*Y/2 + c = 0
    # 2. a*p/q + b*0 + c = 0
    # in integers (a, b, c) yields this:
    a = - q * Y
    c = p * Y
    b = - 2 * p + q * X
    return a, b, c


def apply(line, point):
    a, b, c = line
    x, y = point
    return a * x + b * y + c


def test_line_through():
    fraction = (5, 4)
    point = (0, 1)
    f = tof(fraction)
    line = line_through(fraction, point)
    print(apply(line, (5/4, 0)))
    print(apply(line, point))

# test_line_through() ; exit()


# ss is mutated
def lines_through_point(shape, fraction, ss):
    p, q = fraction
    ordered = np.array(sweep(shape, fraction))
    # TODO line_through could be vectorized and applied to ordered in one go.
    k = len(ordered)
    attempts = 0
    for i in range(k - 1):
        p1, p2 = ordered[i], ordered[i + 1]
        p1ts = p1[0] * q - p, p1[1] * q
        p2ts = p2[0] * q - p, p2[1] * q
        c = cross(p1ts, p2ts)
        assert c <= 0
        if c == 0:
            continue
        attempts += 1

        line = line_between(fraction, p1, p2)
        assert line[0] * p + line[2] * q == 0
        assert apply(line, p1) * apply(line, p2) < 0

        result = slices(line, shape).astype(np.int32)
        result_tup = tuple(map(tuple, result))
        ss.add(result_tup)
    return ss, attempts


# ss = set() ; ss, attempts = lines_through_point(shape, fraction=(5, 4), ss=ss) ; print(ss) ; exit()


# returns a (?, n, m) boolean array of slicing sets,
# namely all the sets whose line intersects the (0, 0) (0, n) side.
# we will get from here to the complete set in collect_lines() by mirroring.
def collect_lines_on_side(shape, centers):
    ss = set()
    total_attempts = 0
    for j, fraction in enumerate(centers):
        ss, attempts = lines_through_point(shape, fraction, ss)
        total_attempts += attempts
        if j % 100 == 0:
            print(j, "/", len(centers), "total_attempts", total_attempts, "collected", len(ss))
    return np.array(list(ss))

# print(collect_lines_on_side(shape, centers).astype(int)) ; exit()


def exact_collect_lines(shape):
    cutpoints = intersect_all_with_side(shape)
    centers = region_centers(cutpoints)

    assert shape[0] == shape[1], "currently only implemented for squares, sorry."
    bitvectors = collect_lines_on_side(shape, centers)
    print("before_mirroring", bitvectors.shape)
    complete = []
    for flip_horiz in (False, True):
        for transpose in (False, True):
            bvs = bitvectors
            if flip_horiz:
                bvs = bvs[:, :, ::-1]
            if transpose:
                bvs = np.swapaxes(bvs, 1, 2)
            complete.append(bvs)
    complete = np.concatenate(complete)
    uniq = np.unique(complete, axis=0)
    print("uniq", uniq.shape)

    # the current naive implementation is too slow.
    # uniq = spernerize(uniq) ; print("subset_filtered", uniq.shape)

    return uniq

# exact_lines = exact_collect_lines(shape, centers).astype(int)


def random_point_of_rect_boundary(shape):
    n, m = shape
    z = np.random.uniform(n + m)
    horizontal = z > n
    if horizontal:
        z -= n
        # if horizontal, then z will be interpreted as second coordinate of shape. if not, as first coordinate.
        x = 0
        y = z
    else:
        x = z
        y = 0
    if np.random.randint(2) == 1:
        x = n - x
    if np.random.randint(2) == 1:
        y = m - y
    return (x, y)


def parametrized_point_of_rect_boundary(shape, z_orig):
    z = z_orig
    n, m = shape
    center_mirror = z > (n + m)
    if center_mirror:
        z -= n + m
    horizontal = z > n
    if horizontal:
        z -= n
        # if horizontal, then z will be interpreted as second coordinate of shape. if not, as first coordinate.
        x = 0
        y = z
    else:
        x = n - z
        y = 0
    if center_mirror:
        x = n - x
        y = m - y
    return x, y


def test_parametrized_point():
    shape = 10, 15
    n, m = shape
    zs = np.linspace(0, (n + m) * 2, 500)
    points = [parametrized_point_of_rect_boundary(shape, z) for z in zs]
    points = np.array(points)
    plt.scatter(points[:, 0], points[:, 1], c=zs, s=zs)
    plt.show()

# test_parametrized_point() ; exit()


def test_random_point():
    shape = 10, 15
    points = [random_point_of_rect_boundary(shape) for _ in range(1000)]
    points = np.array(points)
    plt.scatter(points[:, 0], points[:, 1])
    plt.show()

# test_random_point() ; exit()


def random_line(shape):
    x1, y1 = random_point_of_rect_boundary(shape)
    x2, y2 = random_point_of_rect_boundary(shape)
    a = y1-y2
    b = x2-x1 # reversed!
    c = (y2-y1)*x1-(x2-x1)*y1
    return a, b, c


def parametrized_line(shape, z1, z2):
    x1, y1 = parametrized_point_of_rect_boundary(shape, z1)
    x2, y2 = parametrized_point_of_rect_boundary(shape, z2)
    a = y1-y2
    b = x2-x1 # reversed!
    c = (y2-y1)*x1-(x2-x1)*y1
    return a, b, c


# a line that's almost diagonal except that it intersects cells (x-1, y) and (x+1, y).
# https://www.wolframalpha.com/input/?i=solve+a*x%2B%28y-epsilon%29%2Bc%3D0%2C+a*%28x%2B1%29%2B%28y%2B1%2Bepsilon%29%2Bc%3D0+for+a%2Cc
def line_with_bump(p, upward):
    x, y = p
    N = 1000000
    if upward:
        a = N + 2
        b = N
        c = - x * (N + 2) - N * y - N - 1
    else:
        a = - N - 2
        b = N
        c = 2 * x + 1 + N * (x - y)
    return a, b, c


# line through (p,0) and (0,q)
def crossing_line(p, q):
    b = p
    a = q
    c = - p * q
    return a, b, c

def optimal_half_cross(n, p, q):
    shape = n, n
    agg = np.zeros(shape).astype(int)
    start_line = crossing_line(p, q)
    start_agg = slices(start_line, shape).astype(int)
    agg = start_agg.copy()
    line = list(start_line)
    lines = [line]
    while True:
        line[2] -= (line[0] + line[1]) # parallel translation by (1, 1)
        agg += slices(line, shape).astype(int)
        assert agg.max() == 1
        agg_mirrored = agg[::-1, ::-1]
        lines.append(tuple(line))
        if (agg_mirrored * start_agg).sum() > 0:
            break
    return lines, agg


def lines_to_mask(shape, lines):
    agg = np.zeros(shape).astype(int)
    for line in lines:
        agg += slices(line, shape)
    return (agg > 0).astype(int)


def line_through_floating_point(p1, p2):
    x1, y1 = p1
    x2, y2 = p2
    a = y2 - y1
    b = x1 - x2
    c = (x2 - x1) * y1 - (y2 - y1) * x1
    return a, b, c


def optimal_cross(n, p, q):
    shape = n, n
    lines, agg = optimal_half_cross(n, p, q)
    # print("half cross:") ; print(agg) ; print("now let's finish it")
    size_phase1 = len(lines)
    left = np.nonzero(agg[:, 0])[0][0] - 1, 0
    up = 0, np.nonzero(agg[0, :])[0][0] - 1
    right = np.nonzero(agg[:, -1])[0][-1] + 1, n - 1
    down = n - 1, np.nonzero(agg[-1, :])[0][-1] + 1
    # print(left, up, right, down)
    epsilon = 1e-6
    # the line that barely but intersect cells left and down in their top right corner.
    perpend = line_through_floating_point((left[0] + epsilon, 1 - epsilon), (n - 1 + epsilon, down[1] + 1 - epsilon))
    agg_phase2 = np.zeros_like(agg)
    agg_phase1 = agg.copy()
    step = 0
    while (agg == 0).astype(int).sum() > 0:
        agg += slices(perpend, shape)
        agg_phase2 += slices(perpend, shape) * len(lines)
        lines.append(perpend)
        a, b, c = perpend
        perpend = a, b, c + a - b
        step += 1
        if step > 2 * n:
            print("terminating infinite loop")
            return
    # print(agg_phase2)
    print(len(lines), "=", size_phase1, "+", len(lines)-size_phase1)


def list_optimal_crosses(n):
    epsilon = 1e-6
    for alpha in np.linspace(n // 2 - 10, n // 2 + 10, 100):
        print(alpha)
        optimal_cross(n, alpha, n // 2 + 2 * np.random.uniform())

# n = int(sys.argv[1]) ; list_optimal_crosses(n) ; exit()


def create_general_nontrivial_solution(n):
    assert n % 2 == 0 # for now, but it works for odd n too.
    k = n // 2
    shape = (n, n)
    line0 = line_with_bump((k - 1, k), upward=False)
    line1 = line_with_bump((k, k - 1), upward=False)

    # transpose
    line0 = line0[1], line0[0], line0[2]
    line1 = line1[1], line1[0], line1[2]

    lines = [line0, line1]

    for i in range(2, n - 1):
        line = line_with_bump((i, i - 1), upward=True)
        lines.append(line)

    print("n =", n)
    print("set system size = ", len(lines))
    collected = []
    dump = False
    for i, line in enumerate(lines):
        print("line", i, ":", *line)
        s = slices(line, shape)
        collected.append(s)
        if dump:
            print(s.astype(int))
            print("-----")
    collected = np.array(collected)
    uncovered = (collected.astype(int).sum(axis=0) == 0).astype(int).sum()
    print("uncovered cell count = ", uncovered)
    filename = "exact-%d.npy" % n
    np.save(open(filename, "wb"), collected)
    print("solution saved to", filename)


# n = int(sys.argv[1]) ; create_general_nontrivial_solution(n) ; exit()


def pp(l):
    return str(list(l)).replace('[]', '.').replace('[', '').replace(']', '').replace(' ', '')


def compact_print(ss):
    grid = ss.transpose((1, 2, 0))
    n, m, s = grid.shape
    for i in range(n):
        for j in range(m):
            print(pp(grid[i, j].nonzero()[0]), "\t", end='')
        print()


def create_diagonal_cover_attempt(shape):
    n, m = shape
    assert n == m
    # two close-to-1 slope lines, close to the diagonal,
    # and n-1-2 close-to-minus-1 slope lines, arranged somewhat regularly.
    assert n == 12
    A, B, C, D = 0, n, 2*n, 3*n
    z_pairs = (
        (A+1.5, B+11.8),
        (A+3.5, B+9.4),
        (D+11.2, C+1.7),
        (D+9.5, C+3.7),
        (A+7.1, D+6.5),
        (A+8.8, D+4.5),
        (A+10.8, D+2.7),
        (B+0.7, D+0.3),
        (B+2.7, C+10.3),
        (B+4.7, C+8.3),
        (B+6.7, C+6.3),
    )
    lines = []
    for z1, z2 in z_pairs:
        line = parametrized_line(shape, z1, z2)
        lines.append(line)

    k = 4
    descending = [(-1, intercept) for intercept in np.arange(k) * 2 + 9 + 0.5]
    ascending = [(1, intercept) for intercept in np.arange(n - k - 1) * 2 - 6 + 0.5]
    lines_2par = descending + ascending
    '''
    # this leaves out only a single cell
    sl1 = -1.0
    sl2 = +1.0
    lines_2par = ( # 2 par as in 2 params as in (slope, intercept)
        (sl1, 9.5),
        (sl1, 11.5),
        (sl1, 13.5),
        (sl1, 15.5),

        (sl2, -5.5),
        (sl2, -3.5),
        (sl2, -1.5),
        (sl2, 0.5),
        (sl2, 2.5),
        (sl2, 4.5),
        (sl2, 6.5),
    )
    '''
    lines = tuple((-si[0], 1, -si[1]) for si in lines_2par)

    agg = np.zeros((n, n))
    ss = []
    for line in lines:
        print("slope", - line[0] / line[1], "intercept", - line[2] / line[1])
        result = slices(line, shape)
        ss.append(result)
        agg += result
    print("agg")
    print(agg)
    ss = np.array(ss)
    compact_print(ss)


# create_diagonal_cover_attempt((12, 12)) ; exit()


def create_almost_diagonal_cover_attempt(shape):
    n, m = shape
    assert n == m
    # two close-to-1 slope lines, close to the diagonal,
    # and n-1-2 close-to-minus-1 slope lines, arranged somewhat regularly.
    assert n == 14
    A, B, C, D = 0, n, 2*n, 3*n
    z_pairs = [
        (B+1.75, D-0.9), (B-0.9, D+1.25),
    ]
    # z_pairs += zip(B + np.linspace(5.1, 16.4, 6), B - np.linspace(3.1, 13.1, 6))
    # z_pairs += zip(D + np.linspace(5.1, 14.4, 5), D - np.linspace(3.1, 11.1, 5))
    z_pairs += zip(B + np.linspace(5.2, 16.2, 6), B - np.linspace(3.1, 14.1, 6))
    z_pairs += zip(D + np.linspace(5.2, 15.2, 5), D - np.linspace(3.1, 13.1, 5))

    z_pairs = np.array(z_pairs)

    do_search = True
    if do_search:
        scale = float(sys.argv[1])
        for _ in range(10000):
            z_pairs_perturbed = z_pairs.copy()
            z_pairs_perturbed[2:] += np.random.normal(size=z_pairs_perturbed[2:].shape, scale=scale)
            lines = [parametrized_line(shape, z1, z2) for z1, z2 in z_pairs_perturbed]
            agg = np.zeros((n, n))
            for line in lines:
                result = slices(line, shape)
                agg += result
            print((agg == 0).astype(int).sum())
        return

    lines = []
    for z1, z2 in z_pairs:
        line = parametrized_line(shape, z1, z2)
        lines.append(line)

    agg = np.zeros((n, n))
    ss = []
    for line in lines:
        print("slope", - line[0] / line[1], "intercept", - line[2] / line[1])
        result = slices(line, shape)
        print(result.astype(int))
        ss.append(result)
        agg += result
    print("agg")
    print(agg)
    ss = np.array(ss)
    compact_print(ss)
    print("uncovered", (agg == 0).astype(int).sum())

# create_almost_diagonal_cover_attempt((14, 14)) ; exit()


# minimum score across all covered.
def level(partial_cover, scoring_grid):
    return np.min(np.where(partial_cover, 1e10, 0) + scoring_grid)


def greedy_step_for_almost_diagonal_cover(covered, all_slices, scoring_grid):
    levels = []
    for slc in all_slices:
        lvl = level(covered | slc, scoring_grid)
        levels.append(lvl)
    levels = np.array(levels)
    best_index = np.argmax(levels)
    slc = all_slices[best_index]
    return slc.astype(bool)


def greedy_search_for_almost_diagonal_cover(shape):
    n, m = shape
    assert n == m
    # two close-to-1 slope lines, close to the diagonal,
    # and n-1-2 close-to-minus-1 slope lines, arranged somewhat regularly.

    filename = "set-systems/%d-%d.npy" % shape
    try:
        f = open(filename, "rb")
        all_slices = np.load(f)
        print("took set system from cache %s" % filename)
    except:
        print("%s not found" % filename)
        exit()


    A, B, C, D = 0, n, 2*n, 3*n
    z_pairs = [
        (B+1.01, D-0.99), (B-0.99, D+1.01),
    ]
    z_pairs = np.array(z_pairs)
    covered = np.zeros((n, n)).astype(bool)
    solution = []
    for z1, z2 in z_pairs:
        line = parametrized_line(shape, z1, z2)
        result = slices(line, shape)
        solution.append(result)
        covered |= result.astype(bool)

    scoring_grid = np.arange(n)[:, np.newaxis] + np.arange(n)[np.newaxis, :]
    while not np.all(covered):
        slc = greedy_step_for_almost_diagonal_cover(covered, all_slices, scoring_grid)
        covered |= slc
        solution.append(slc)
        '''
        print("=====")
        print(slc.astype(int))
        print("-----")
        print(covered.astype(int))
        '''
        print(len(solution))
        sys.stdout.flush()
    solution = np.array(solution)
    print(solution.shape)
    print("found solution of size %d for n=%d" % (len(solution), n))
    np.save(open("greedy.%d.npy" % n, "wb"), solution)


# n = int(sys.argv[1]) ; greedy_search_for_almost_diagonal_cover((n, n)) ; exit()


def search_for_diagonal_covers(shape):
    n, m = shape
    k = 4
    descending = [(-1, intercept) for intercept in np.arange(k) * 2 + 9 + 0.5]
    ascending = [(1, intercept) for intercept in np.arange(n - k - 1) * 2 - 6 + 0.5]
    lines_2par = np.array(descending + ascending)

    def perturb(lines):
        lines = lines.copy()
        lines[:, 0] += np.random.normal(scale=0.05, size=1) # len(lines))
        lines[:, 1] *= 1 + np.random.normal(scale=0.1, size=1)
        lines[:, 1] += np.random.normal(scale=0.5, size=1) # len(lines))
        return lines

    for _ in range(10000):
        lines = tuple((-si[0], 1, -si[1]) for si in perturb(lines_2par))

        agg = np.zeros((n, n))
        for line in lines:
            result = slices(line, shape)
            agg += result
        # print(agg) ; print("============================")
        if agg.min() > 0:
            print("VICTORY", agg)
            exit()

# search_for_diagonal_covers((12, 12)) ; exit()


def visualize_partition(shape, granularity):
    n, m = shape
    zs = np.linspace(0, (n + m) * 2, granularity)
    surface = np.zeros((granularity, granularity))
    for i1, z1 in enumerate(zs):
        for i2, z2 in enumerate(zs):
            line = parametrized_line(shape, z1, z2)
            result = slices(line, shape)

            if np.sum(result.astype(int)) >= 1:
                result_tup = tuple(map(tuple, result))
                surface[i1, i2] = hash(result_tup)
        if i1 % 100 == 0:
            print(i1)
    plt.figure(figsize=(20, 20))
    plt.imshow(surface)
    plt.xticks([])
    plt.yticks([])
    plt.savefig("mobius.%d-%d.png" % shape)

# visualize_partition(shape=(5, 5), granularity=1000) ; exit()


# do at most samples attempts, but halt prematurely if there's no new find in the last patience attempts.
def sampling_collect_lines(shape, samples, patience):
    ss = set()
    total_attempts = 0
    attempts = 0
    while True:
        line = random_line(shape)
        if line[0] == 0.0 or line[1] == 0.0:
            # these lines are the result of sampling the two endpoints from the same side. they do not help.
            continue
        total_attempts += 1
        attempts += 1
        result = slices(line, shape)
        if np.sum(result.astype(int)) >= 1:
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


def totuple(a):
    try:
        return tuple(totuple(i) for i in a)
    except TypeError:
        return a


def parametrized_collect_lines(shape, granularity):
    ss = set()
    n, m = shape
    zs = np.linspace(0, (n + m) * 2, granularity)
    total_attempts = 0
    for i1, z1 in enumerate(zs):
        for i2, z2 in enumerate(zs):
            if i2 >= i1:
                continue
            total_attempts += 1
            line = parametrized_line(shape, z1, z2)
            result = slices(line, shape)

            if np.sum(result.astype(int)) >= 1:
                result_tup = tuple(map(tuple, result))
                ss.add(result_tup)
            if total_attempts % 10000 == 0:
                print(total_attempts, len(ss))
    return ss


# pick endpoint 1 from ((0, 0), (0, n * ratio))
# pick endpoint 2 from ((n, n), (n * (1-ratio), n)
def sampling_collect_diagonal_lines(shape, samples):
    ss = set()
    n, m = shape
    assert n == m
    total_attempts = 0
    for i in range(samples):
        # this worked for n=11, 12. did not work for n=13.
        # found n=11 nontrivial solution in 500 maxiters, 9 mins. on spernerized set system.
        # found n=12 nontrivial solution in 750 maxiters, 33 mins. on spernerized set system.
        # have not found n=13 nontrivial solution in 10000 maxiters, 300 mins. no spernerization there
        z1 = np.random.uniform(low=n * 0.4, high=n + n * 0.1)
        z2 = 4 * n - z1 + np.random.uniform(low=-n * 0.2, high=n * 0.2)

        total_attempts += 1
        line = parametrized_line(shape, z1, z2)
        result = slices(line, shape)

        if np.sum(result.astype(int)) >= 1:
            result_tup = tuple(map(tuple, result))
            before = len(ss)
            ss.add(result_tup)
            if len(ss) > before:
                pass
                # print(pretty(result)) ; print()
        if total_attempts % 10000 == 0:
            print(total_attempts, len(ss))

    ss = np.array(list(ss))
    complete = []
    print("before mirroring", len(ss))
    for flip_horiz in (False, True):
        for flip_vert in (False, True):
            bvs = ss
            if flip_horiz:
                bvs = bvs[:, :, ::-1]
            if flip_vert:
                bvs = bvs[:, ::-1, :]
            complete.append(bvs)
    complete = np.concatenate(complete)
    print("after mirroring", len(complete))
    uniq = np.unique(complete, axis=0)
    print("after unique", len(uniq))
    print(uniq.sum(axis=0))
    do_spernerization = False
    if do_spernerization:
        uniq = spernerize(uniq)
        print("after spernerization", len(uniq))
    return uniq


# n = int(sys.argv[1])
# shape = n, n
# ss = parametrized_collect_diagonal_lines(shape, ratio=0.6, samples=100000) ; print(n, len(ss)) ; exit()


# https://stackoverflow.com/a/14107790/383313
def spernerize(set_system):
    shape = set_system.shape
    ss = set_system.reshape((len(set_system), -1))
    kept = []
    for line in ss:
        elements = line.nonzero()[0]
        # intersection of all columns corresponding to the elements of row 'line'
        summary = np.min(ss[:, elements], axis=1)
        # 'line' is by definition an element of this intersection.
        # if there are more, line is not kept.
        if summary.sum() == 1:
            kept.append(line)
    kept = np.array(kept).reshape((-1, shape[1], shape[2]))
    return kept


# nr_of_lines=2 (-1, +1)
# nr_of_lines=3  (-2, 0, 2)
def create_waist(shape, nr_of_lines):
    agg = np.zeros(shape, dtype=int)
    for c in range(-nr_of_lines+1, nr_of_lines, 2):
        line = (1, -1, c - 0.5)
        s = slices(line, shape)
        # print(pretty(s))
        agg += s.astype(int)
    return agg


def create_irregular_waist(shape):

    # this works for n=11, found a 10-line solution in 100 niters.
    waist_lines = [(1.0, -1.1, -0.7), (1.0, -1.1, 1.4), (1.0, -1.1, -2.8), (1.0, -1.1, 3.48)]

    # tried this for n=12, but it didn't work.
    # waist_lines = [(1.0, -1.1, -0.7), (1.0, -1.1, 1.4), (1.0, -1.1, -2.8), (1.0, -1.1, 3.45), (1.0, -1.1, -4.85), (1.0, -1.1, -5.7)]
    # waist_lines = [(1.0, -1.1, -0.7), (1.0, -1.1, 1.4), (1.0, -1.1, -2.8), (1.0, -1.1, 3.45), (1.0, -1.1, -4.87)]

    agg = np.zeros(shape, dtype=int)
    for line in waist_lines:
        s = slices(line, shape)
        print(pretty(s))
        agg += s.astype(int)
    return (agg > 0).astype(int), len(waist_lines)


def build_set_system(shape, samples, patience, waist=None):
    filename = "%d-%d.npy" % shape # I put "%d-%d-mono.npy" % shape here when working with the generalized lines.
    try:
        f = open(filename, "rb")
        collected_slices = np.load(f)
        print("took set system from cache %s" % filename)
    except OSError:
        # cs = exact_collect_lines(shape)
        # cs = sampling_collect_diagonal_lines(shape, samples=samples)
        cs = sampling_collect_lines(shape, samples=samples, patience=patience)
        # cs = parametrized_collect_lines(shape, granularity=1000)

        # this is the version with continuous monotone "lines", see vis_monotone.py for a bit more detail.
        # cs = sample_monotones(shape, sample_count=1000000) ; cs = np.concatenate([cs, cs[:, ::-1]])

        collected_slices = np.array(list(cs))
        np.save(open(filename, "wb"), collected_slices)

    if waist is not None:
        collected_slices = list(collected_slices)
        collected_slices.append(waist)
        collected_slices = np.array(collected_slices)
        print("added waist")

    ss = collected_slices
    ss = ss.reshape((len(ss), -1)) # flatten the nxm grids to n*m vectors
    ss = ss.T
    # now ss.shape == (number of points, number of sets).
    cost = np.ones((ss.shape[-1]), dtype=np.int64)
    return collected_slices, ss, cost


def precreate_set_system_caches():
    for n in range(2, 100):
        collected_slices, ss, cost = build_set_system((n, n), samples=None, patience=None, waist=None)

# precreate_set_system_caches() ; exit()


def visualize_solution(ss):
    ss =  np.array(ss)
    ss_size, n, m = ss.shape
    d = 100
    N, M = d * n + 1, d * m + 1
    from PIL import Image, ImageDraw
    
    im = Image.new('RGB', (N, M), color='white')
    draw = ImageDraw.Draw(im)
    for i in range(n + 1):
        draw.line((i * d, 0, i * d, M), fill='blue')
    for i in range(m + 1):
        draw.line((0, i * d, N, i * d), fill='blue')
    im.save("img.png")


n = int(sys.argv[1])
shape = n, n

n, m = shape
samples = 100000
patience = 10000
maxiters = 10000 # unlike the upper bound which needs luck, the lower bound normally converges after maxiters=2.

# unfortunately it seem like even nr_of_waist_lines=2 (that is, forcing the two middle diagonals into the cover)
# prevents the optimizer from finding an n-1 solution, probably because it does not even exist with this restriction.
do_waist_hack = False
if do_waist_hack:
    # nr_of_waist_lines = 6 ; waist = create_waist(shape, nr_of_waist_lines)
    waist, nr_of_waist_lines = create_irregular_waist(shape)
    print(pretty(waist))
    print("adding a single set to the system that is worth %d lines" % nr_of_waist_lines)
    print("please add %d to the upper bound manually" % (nr_of_waist_lines - 1))
    print("elements covered by waist:", waist.sum())
else:
    waist = None

collected_slices, ss, cost = build_set_system((n, m), samples=samples, patience=patience, waist=waist)
print(collected_slices.shape, ss.shape, cost.shape)


# interpolates between a quadratic polynomial and a rotationally symmetric formula.
def create_lagrangian(shape, weight=None):
    g = mygrid((n-1, m-1)).astype(float) # reusing a code for cells that was created for crossings. (as in chess vs go)
    x = g[:, :, 0] / (n-1) * 2 - 1
    y = g[:, :, 1] / (m-1) * 2 - 1
    dist = np.sqrt(x * x + y * y)
    s2 = 2 ** 0.5

    xa = np.abs(x)
    ya = np.abs(y)

    # the good old fitted polynomial
    lag = x * x + y * y - 2 * x * x * y * y
    lag /= lag.sum()
    return lag


    # fitted a degree 6 poly of abs(x), abs(y)
    # to an empirical Lagrangian found for (n=100, 40000 sampled lines)
    # with the nonparametric version of optimize.py.
    # got 0.7015n when evaluating on the (n=100, 100000 sampled lines) set system.
    coeffs = np.array([ 4.67821221e-02, 1.54725799e+00, -2.05429441e+01, 1.07163134e+02,
     -2.34282340e+02, 2.31279650e+02, -8.42997020e+01, 1.54725799e+00,
     -4.86969904e+01, 5.64970506e+02, -2.46279444e+03, 4.84154366e+03,
     -4.39738317e+03, 1.50209961e+03, -2.05429441e+01, 5.64970506e+02,
     -6.57743065e+03, 2.88418628e+04, -5.64993903e+04, 5.08578914e+04,
     -1.71773480e+04, 1.07163134e+02, -2.46279444e+03, 2.88418628e+04,
     -1.29274313e+05, 2.57385452e+05, -2.34241721e+05, 7.96864840e+04,
     -2.34282340e+02, 4.84154366e+03, -5.64993902e+04, 2.57385452e+05,
     -5.19336442e+05, 4.77142322e+05, -1.63389834e+05, 2.31279650e+02,
     -4.39738317e+03, 5.08578914e+04, -2.34241721e+05, 4.77142322e+05,
     -4.41211928e+05, 1.51700150e+05, -8.42997020e+01, 1.50209961e+03,
     -1.71773480e+04, 7.96864840e+04, -1.63389834e+05, 1.51700150e+05,
     -5.22615240e+04])
    print(coeffs.reshape((7, 7)))
    xy_monom = np.array([xa.flatten() ** i * ya.flatten() ** j for i in range(7) for j in range(7)])
    lag = xy_monom.T.dot(coeffs).reshape((n, n))
    return lag


    # that's the same approach as the degree 6, just degree 2, gets to 0.6776n.
    lag = 0.09928857 - 0.24501929 * xa + 1.15962901 * xa*xa - 0.24501929 * ya + 2.06339929 * xa*ya - 1.26814885 * xa*xa*ya + 1.15962901 * ya*ya - 1.26814885 * xa*ya*ya - 1.58775847 * xa*xa*ya*ya
    return lag


    # rotationally symmetric formula based on histogram of distance-value scatterplot:
    lag1 = (dist <= 1) * dist + (dist > 1) * (dist - s2) / (1 - s2)
    lag1 /= lag1.sum()

    # the good old fitted polynomial
    lag2 = x * x + y * y - 2 * x * x * y * y
    lag2 /= lag2.sum()

    lag = lag1 * weight + lag2 * (1 - weight)
    return lag


# update: this does not work, gives a cca 0.5n lower bound.
def create_gergo_lagrangian(shape):
    n, m = shape
    assert n == m
    g = mygrid((n-1, m-1)).astype(float) # reusing a code for cells that was created for crossings. (as in chess vs go)
    x = g[:, :, 0] / (n-1) * 2 - 1
    y = g[:, :, 1] / (m-1) * 2 - 1
    l1_dist = np.abs(x) + np.abs(y)
    diarect = (np.abs(x - y) <= 0.5) & (np.abs(x + y) <= 1.5)
    diarect = diarect.astype(float)
    complement = (diarect + diarect[:, ::-1] == 0) & (np.abs(x + y) <= 1.5)
    complement = complement.astype(float)
    complement /= 2
    cross = np.abs(diarect - diarect[:, ::-1])
    print(cross + complement)
    return cross + complement


def create_daniel_lagrangian(shape, weights):
    n, m = shape
    assert n == m
    g = mygrid((n-1, m-1)) # reusing a code for cells that was created for crossings. (as in chess vs go)
    x = g[:, :, 0] - (n - 1) // 2
    y = g[:, :, 1] - (n - 1) // 2
    linf = np.maximum(np.abs(x), np.abs(y))
    lag = np.zeros(shape)
    for i, w in enumerate(weights):
        lag += (linf == i) * w
    return lag


def lagrangian_to_lower_bound(lagrangian, set_system):
    dual_covers = lagrangian.flatten().dot(set_system.astype(int))
    # for each slice, the sum of the lagrangian at its elements
    # supposed to be smaller than 1, but here the lagrangian is unnormalized yet.
    worst = max(dual_covers)
    worst_index = np.argmax(dual_covers)
    worst_line = set_system[:, worst_index].reshape(shape).astype(int)
    '''
    for row in worst_line:
        print(row)
    '''
    print("worst", worst, "sum", np.sum(lagrangian))
    dual_covers /= worst
    lb = np.sum(lagrangian) / worst
    return lb

    # plt.hist(dual_covers, bins=30)
    # plt.show()


def bad_line_test():
    n, m = shape
    assert n % 3 == 0 and n % 2 == 1
    k = n // 2 + 1
    l = int(n / 6 + 1)
    weights = [0] * l + [1] * (k-l)
    lagrangian = create_daniel_lagrangian(shape, weights)
    print(lagrangian_to_lower_bound(lagrangian, ss), weights)

# bad_line_test() ; exit()


def tune_mixing_weight():
    interval = np.linspace(0, 1, 2) ** 2
    n, m = shape
    assert n % 2 == 1
    k = n // 2 + 1
    weight_vecs = np.array(np.meshgrid(*([interval] * k))).T.reshape(-1, k) # direct product of k intervals
    weight_vecs = [[0] * l + [1] * (k-l) for l in range(1, k)]
    print(weight_vecs)
    print("len(weight_vecs)", len(weight_vecs), file=sys.stderr)
    for i, weights in enumerate(weight_vecs):
        lagrangian = create_daniel_lagrangian(shape, weights)
        print(lagrangian_to_lower_bound(lagrangian, ss), weights)
        if i % 10000 == 0:
            print(i, file=sys.stderr)

# tune_mixing_weight() ; exit()

print("evaluating analytical lagrangian") ; analytical_lagrangian = create_lagrangian(shape) ; print("lb", lagrangian_to_lower_bound(analytical_lagrangian, ss))

# cached_lagrangian = np.load(open("lagrangian.%d-%d.npy" % shape, "rb")) ; print("lb", lagrangian_to_lower_bound(cached_lagrangian, ss))


def to_setcoverpy_input(collected_slices):
    ss = collected_slices
    ss = ss.reshape((len(ss), -1)) # flatten the nxm grids to n*m vectors
    ss = ss.T
    # now ss.shape == (number of points, number of sets).
    cost = np.ones((ss.shape[-1]), dtype=np.int64)
    return ss, cost

def main_batch(collected_slices, maxiters):
    shape = collected_slices.shape[1:]
    n, m = shape
    assert n == m
    ss, cost = to_setcoverpy_input(collected_slices)
    found = 0
    for i in range(10000):
        g = setcover.SetCover(ss, cost, maxiters=maxiters)
        solution, time_used = g.SolveSCP()
        bitvec = g.s
        solution = collected_slices[bitvec, :]
        if len(solution) < n: # nontrivial solution
            found += 1
            filename = "solution.%d.%05d.npy" % (n, abs(hash(totuple(solution))) % 100000)
            print("found %s. nontrivial solution, saving it to %s" % (found + 1, filename))
            with open(filename, "wb") as f:
                np.save(f, solution)
        print("%d. set cover restart, %d solutions so far." % (i, found))

main_batch(collected_slices, maxiters=100) ; exit()


def main_interactive(collected_slices, maxiters):
    shape = collected_slices.shape[1:]
    n, m = shape
    ss, cost = to_setcoverpy_input(collected_slices)

    g = setcover.SetCover(ss, cost, maxiters=maxiters)
    print("starting set cover solver")
    solution, time_used = g.SolveSCP()

    bitvec = g.s
    solution = collected_slices[bitvec, :].reshape((-1, n, m))
    np.save(open("solution.%d-%d.npy" % shape, "wb"), solution)

    agg = np.zeros((n, m), dtype=int)
    for i, s in enumerate(solution):
        print(i)
        print(pretty(s))
        agg += s.astype(int)

    print("aggregate")
    print(agg)

    lagrangian = np.array(g.u).reshape((n, m))
    np.save(open("lagrangian.%d-%d.npy" % shape, "wb"), lagrangian)
    plt.imshow(lagrangian)
    plt.savefig("vis.png")
    plt.clf()

    # plt.hist(lagrangian.flatten().dot(ss.astype(int)), bins=30)
    # plt.show()


main_interactive(collected_slices, maxiters)
