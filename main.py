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
    # TODO don't use floating point
    f = tof(fraction)

    # unused, just for reference
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
    shape = 50, 52
    fraction = 25, 1
    ordered = np.array(sweep(shape, fraction))
    plt.scatter(ordered[:, 0], ordered[:, 1], c=range(len(ordered)))
    plt.show()

# test_sweep()



def line_through(fraction, point):
    p, q = fraction # interpreted as point (p/q, 0)
    x, y = point
    # solving
    # 1. a*x + b*y + c = 0
    # 2. a*p/q + b*0 + c = 0
    # in integers (a, b, c) yields this:
    a = q * y
    c = p * y
    b = - p - q * x
    return a, b, c


shape = 5, 5
cutpoints = intersect_all_with_side(shape)
centers = region_centers(cutpoints)
# print(centers)

# returns a (?, n, m) boolean array of slicing sets,
# namely all the sets whose line intersects the (0, 0) (0, n) side.
# we will get from here to the complete set in collect_lines() by mirroring.
def collect_lines_on_side(shape, centers):
    print("collect_lines_on_side() still buggy, seems only to collect downward-sloping lines")
    ss = set()
    total_attempts = 0
    reasonable_attempts = 0
    for j, fraction in enumerate(centers):
        p, q = fraction
        ordered = np.array(sweep(shape, fraction))
        # TODO line_through could be vectorized and applied to ordered in one go.
        k = len(ordered)
        for i in range(k - 1):
            p1, p2 = ordered[i], ordered[i + 1]
            p1ts = p1[0] * q - p, p1[1] * q
            p2ts = p2[0] * q - p, p2[1] * q
            c = cross(p1ts, p2ts)
            assert c <= 0
            total_attempts += 1
            if c == 0:
                continue
            reasonable_attempts += 1
            # find the two normal equations and sum them:
            line1 = line_through(fraction, p1)
            line2 = line_through(fraction, p2)
            line = np.array(line1) + np.array(line2)
            result = slices(line, shape)
            result_tup = tuple(map(tuple, result))
            ss.add(result_tup)
        if j % 1000 == 0:
            print(total_attempts, reasonable_attempts, len(ss))
    return np.array(list(ss))


def collect_lines(shape, centers):
    assert shape[0] == shape[1], "currently only implemented for squares, sorry."
    bitvectors = collect_lines_on_side(shape, centers)
    print(bitvectors.shape, "bitvectors.shape")
    complete = []
    for flip_horiz in (False, True):
        bvs = bitvectors
        if flip_horiz:
            bvs = bvs[:, :, ::-1]
        for flip_vert in (False, True):
            if flip_vert:
                bvs = bvs[:, ::-1, :]
            for transpose in (False, True):
                if transpose:
                    bvs = np.swapaxes(bvs, 1, 2)
                complete.append(bvs)
    complete = np.concatenate(complete)
    print(complete.shape)
    uniq = np.unique(complete, axis=0)
    print("uniq", uniq.shape)
    return uniq

    for bv in bitvectors:
        print(pretty(bv))
        print()
    print("====")
    vert_mirrored = bitvectors[:, ::-1, :]
    horiz_mirrored = bitvectors[:, :, ::-1]
    print()
    print(pretty(horiz_mirrored[0]))
    collected = np.concatenate([bitvectors, horiz_mirrored])
    print(collected.shape, "collected.shape")
    uniq = np.unique(collected, axis=0)
    print("uniq", uniq.shape)
    return collected

lines = collect_lines(shape, centers)
print("lines, exact, maybe buggy", lines.shape)


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
    horiz_mirror = z > (n + m)
    if horiz_mirror:
        z -= n + m
    vert_mirror = z > (n + m) / 2
    if vert_mirror:
        z -= (n + m) / 2
    horizontal = z > n / 2
    if horizontal:
        z -= n / 2
        # if horizontal, then z will be interpreted as second coordinate of shape. if not, as first coordinate.
        x = 0
        y = z
    else:
        x = z
        y = 0
    if horiz_mirror:
        x = n - x
    if vert_mirror:
        y = m - y
    return x, y


def test_parametrized_point():
    shape = 10, 15
    n, m = shape
    zs = np.linspace(0, (n + m) * 2, 500)
    points = [parametrized_point_of_rect_boundary(shape, z) for z in zs]
    points = np.array(points)
    plt.scatter(points[:, 0], points[:, 1], c=zs)
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


# random_line((10, 15)) ; exit()

# do at most samples attempts, but halt prematurely if there's no new find in the last patience attempts.
def collect_slices(shape, samples, patience):
    ss = set()
    total_attempts = 0
    attempts = 0
    while True:
        total_attempts += 1
        attempts += 1
        line = random_line(shape)
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
    return ss


def parametrized_collect_slices(shape, granularity):
    ss = set()
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


# nr_of_lines=2 (-1, +1)
# nr_of_lines=3  (-2, 0, 2)
def create_waist(shape, nr_of_lines):
    waist_slices = []
    agg = np.zeros(shape, dtype=int)
    for c in range(-nr_of_lines+1, nr_of_lines, 2):
        line = (1, -1, c - 0.5)
        s = slices(line, shape)
        # print(pretty(s))
        agg += s.astype(int)
    return agg


def build_set_system(shape, samples, patience, waist=None):
    filename = "%d-%d.npy" % shape
    try:
        f = open(filename, "rb")
        collected_slices = np.load(f)
        print("took set system from cache %s" % filename)
    except OSError:
        cs = collect_slices(shape, samples=samples, patience=patience)
        # cs = parametrized_collect_slices(shape, granularity=1000)
        collected_slices = np.array(list(cs))
        np.save(open("%d-%d.npy" % shape, "wb"), collected_slices)

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


# shape = 10, 10

n, m = shape
samples = 200000
patience = 10000
maxiters = 200 # unlike the upper bound which needs luck, the lower bound normally converges after maxiters=2.

# unfortunately it seem like even nr_of_waist_lines=2 (that is, forcing the two middle diagonals into the cover)
# prevents the optimizer from finding an n-1 solution, probably because it does not even exist with this restriction.
do_waist_hack = False
if do_waist_hack:
    nr_of_waist_lines = 2
    print("adding a single set to the system that is worth %d lines" % nr_of_waist_lines)
    print("please add %d to the upper bound manually" % (nr_of_waist_lines - 1))
    waist = create_waist(shape, nr_of_waist_lines)
    print(pretty(waist))
else:
    waist = None

collected_slices, ss, cost = build_set_system((n, m), samples=samples, patience=patience, waist=waist)
print(collected_slices.shape, ss.shape, cost.shape)


# interpolates between a quadratic polynomial and a rotationally symmetric formula.
def create_lagrangian(shape, weight):
    g = mygrid((n-1, m-1)).astype(float) # reusing a code for cells that was created for crossings. (as in chess vs go)
    x = g[:, :, 0] / (n-1) * 2 - 1
    y = g[:, :, 1] / (m-1) * 2 - 1
    dist = np.sqrt(x * x + y * y)
    s2 = 2 ** 0.5

    # rotationally symmetric formula based on histogram of distance-value scatterplot:
    lag1 = (dist <= 1) * dist + (dist > 1) * (dist - s2) / (1 - s2)
    lag1 /= lag1.sum()

    # fitted polynomial
    lag2 = x * x + y * y - 2 * x * x * y * y
    lag2 /= lag2.sum()

    lag = lag1 * weight + lag2 * (1 - weight)
    return lag


def lagrangian_to_lower_bound(lagrangian, set_system):
    dual_covers = lagrangian.flatten().dot(set_system.astype(int))
    # for each slice, the sum of the lagrangian at its elements
    # supposed to be smaller than 1, but here the lagrangian is unnormalized yet.
    worst = max(dual_covers)
    lb = np.sum(lagrangian) / worst
    print("lower bound", lb)

    # plt.hist(dual_covers * np.sum(lagrangian) / worst, bins=30)
    # plt.show()


def tune_mixing_weight():
    for weight in np.linspace(0, 1, 20):
        lagrangian = create_lagrangian(shape, weight)
        print(weight)
        lagrangian_to_lower_bound(lagrangian, ss)


# print("evaluating analytical lagrangian") ; analytical_lagrangian = create_lagrangian(shape, 1) ; lagrangian_to_lower_bound(analytical_lagrangian, ss)

# cached_lagrangian = np.load(open("lagrangian.%d-%d.npy" % shape, "rb")) ; lagrangian_to_lower_bound(cached_lagrangian, ss)



g = setcover.SetCover(ss, cost, maxiters=maxiters)
print("starting set cover solver")
solution, time_used = g.SolveSCP()
bitvec = g.s
solution = collected_slices[bitvec, :].reshape((-1, n, m))
agg = np.zeros((n, m), dtype=int)
for i, s in enumerate(solution):
    # print(i)
    # print(pretty(s))
    agg += s.astype(int)

print("aggregate")
print(agg)


lagrangian = np.array(g.u).reshape((n, m))
np.save(open("lagrangian.%d-%d.npy" % shape, "wb"), lagrangian)
plt.imshow(lagrangian)
plt.savefig("vis.png")
plt.clf()

plt.hist(lagrangian.flatten().dot(ss.astype(int)), bins=30)
plt.show()
