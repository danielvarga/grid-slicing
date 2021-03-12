import sys
from collections import defaultdict
import numpy as np
import SetCoverPy.setcover as setcover

import matplotlib.pyplot as plt


# the next two functions are part of an abandoned attempt to
# find all slices where the line goes through a specific point.
# the idea was that we draw lines between all point pairs, intersect these
# lines with a bounding circle, and this splits the circle into arcs.
# we then pick a point on each arc, and run this search.

def mygrid(shape):
    n, m = shape
    xi = np.arange(n + 1)
    yi = np.arange(m + 1)
    xx, yy = np.meshgrid(xi, yi)
    # g = np.stack([np.arange(n + 1)[:, np.newaxis], np.arange(m + 1)[np.newaxis, :]], axis=-1)
    g = np.stack([xx, yy], axis=-1)
    return g


def view_from_point(shape, p):
    x, y = p
    # todo move creation out of hot path
    g = mygrid(shape)
    f = g.astype(float).reshape((-1, 2)) # f as in flat, as in more flat than g
    f[:, 0] -= x
    f[:, 1] -= y
    angles = np.arctan2(f[:, 0], f[:, 1])
    perm = np.argsort(angles)
    ordered = f[angles]

# view_from_point((6, 7), (4.1, 5.1)) ; exit()


# get lines for each pair of grid points, and intersect these lines with the big rectangle boundary.
# use integer arithmetic as long as possible.
def all_pairs_lines(shape):
    g = mygrid(shape)
    f = g.reshape((-1, 2))
    lines = set()
    for i1, (x1, y1) in enumerate(f):
        for i2, (x2, y2) in enumerate(f):
            if i2 <= i1:
                continue
            a = y1-y2
            b = x2-x1 # reversed!
            c = (y2-y1)*x1-(x2-x1)*y1
            g = np.gcd.reduce([a, b, c])
            a //= g
            b //= g
            c //= g
            # in principle sign is unnormalized, can coincide. in practice it can't because p1 < p2 lexicographically.
            lines.add((a, b, c))
    return lines


# does frac[0]/frac[1] intersect [0, n]?
def rational_intersection(frac, n):
    a, b = frac
    assert b > 0
    return (b != 0) and (a * b >= 0) and (a <= b*n)


def normalize(frac):
    a, b = frac
    assert b != 0
    if b < 0:
        a, b = -a, -b
    g = np.gcd(a, b)
    return a // g, b // g


# returns the intersection of line and big box boundary,
# in format ((x_divident, x_divisor), (y_divident, y_divisor))
def line_box_intersect(shape, line):
    a, b, c = line
    n, m = shape
    # intersection is when a*x+b*y+c==0

    ps = []
    # case x = 0
    if b !=0:
        frac = normalize((-c, b))
        if rational_intersection(frac, m):
            ps.append(((0, 1), frac))
    # case x = n
    if b!=0:
        frac = normalize((-c - a*n, b))
        if rational_intersection(frac, m):
            ps.append(((n, 1), frac))
    # case y = 0
    if a!=0:
        frac = normalize((-c, a))
        if rational_intersection(frac, n):
            ps.append((frac, (0, 1)))
    # case y = m
    if a!=0:
        frac = normalize((-c - b*m, a))
        if rational_intersection(frac, n):
            ps.append((frac, (m, 1)))
    return ps

def test_line_box_intersect():
    shape = 10, 15
    line = 1, -1, 0
    print(line_box_intersect(shape, line))
    line = -1, 1, 10
    print(line_box_intersect(shape, line))


# test_line_box_intersect() ; exit()


def test_all_pairs():
    shape = 11, 11
    n, m = shape
    lines = np.array(list(all_pairs_lines((n, m))))
    print("%d lines collected" % len(lines))
    points = set()
    for line in lines:
        ps = line_box_intersect(shape, line)
        points |= set(ps)
    print(len(points))
    plt.scatter([xup/xdown for ((xup, xdown), (yup, ydown)) in points], [yup/ydown for ((xup, xdown), (yup, ydown)) in points])
    plt.show()


# test_all_pairs() ; exit()


def collect_boundary_points(shape):
    n, m = shape
    print("shape %d, %d" % (n, m))
    lines = np.array(list(all_pairs_lines((n, m))))
    print("%d lines collected" % len(lines))
    points = set()
    for line in lines:
        ps = line_box_intersect(shape, line)
        points |= set(ps)
    # seems like number of different boundary points is approximately n^3.
    print("%d boundary points collected" % len(points))
    return points


from functools import cmp_to_key

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


def average_fractions(x, y):
    a, b = x # a/b
    c, d = y # c/d
    # a/b + c/d = (ad + cb) / 2bd
    return normalize((a * d + c * b, 2 * b * d))


def test_average_fractions():
    for _ in range(1000):
        a = np.random.randint(low=-10, high=10, size=(2, 2))
        if a[0, 1] == 0 or a[1, 1] == 0:
            continue
        x = normalize(a[0, :])
        y = normalize(a[1, :])
        avg = average_fractions(x, y)
        assert np.isclose(tof(avg), (tof(x) + tof(y)) / 2)

# test_average_fractions() ; exit()

# input: a set of points on the boundary of the box, in fractional format.
# removing this set from the boundary splits it into open line segments.
# output: a point from each segment.
# it is assumed that the box corners are in the input.
def boundary_region_centers(boundary_points):
    bs = [(xup, xdown) for ((xup, xdown), (yup, ydown)) in boundary_points if yup == 0]
    print("identified %d boundary points on single side out of %d" % (len(bs), len(boundary_points)))
    bs = sort_fractions(bs)
    centers = []
    for i in range(len(bs) - 1):
        centers.append(average_fractions(bs[i], bs[i + 1]))
    raise "todo put back on 2d plane, todo do it for all sides, employing symmetry"
    return centers


shape = 20, 20
boundary_points = collect_boundary_points(shape)
centers = boundary_region_centers(boundary_points)


exit()

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


def pretty(s):
    return str(s.astype(int)).replace('1', '■').replace('0', '·').replace('[', ' ').replace(']', ' ').replace(' ', '')

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


shape = 8, 8

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
