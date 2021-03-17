import sys
import numpy as np
import matplotlib.pyplot as plt

n = int(sys.argv[1])
m = n
shape = n, m


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


def parametrized_line(shape, z1, z2):
    x1, y1 = parametrized_point_of_rect_boundary(shape, z1)
    x2, y2 = parametrized_point_of_rect_boundary(shape, z2)
    a = y1-y2
    b = x2-x1 # reversed!
    c = (y2-y1)*x1-(x2-x1)*y1
    return a, b, c

# input is (-1, y1) to (+1, y2) in plusminus one coorddinate system if opposite,
# else input is (-1, y1) to (y2, -1) or whichever mirror of it.
# output is the line, but now in the usual (0, n) c. system.
def parametrized_line_on_opposite(shape, y1, y2, opposite):
    n, m = shape
    assert n == m
    z1 = (y1 + 1) / 2 * n
    if opposite:
        z2 = 3 * n - (y2 + 1) / 2 * n
    else:
        z2 = 2 * n + (y2 + 1) / 2 * n
    return parametrized_line(shape, z1, z2)


def mygrid(shape):
    n, m = shape
    xi = np.arange(n + 1)
    yi = np.arange(m + 1)
    xx, yy = np.meshgrid(xi, yi)
    # g = np.stack([np.arange(n + 1)[:, np.newaxis], np.arange(m + 1)[np.newaxis, :]], axis=-1)
    g = np.stack([xx, yy], axis=-1)
    return g


def vis_quadratic(samples, opposite):
    m = n = samples
    g = mygrid((n-1, m-1)).astype(float) # reusing a code for cells that was created for crossings. (as in chess vs go)
    y1 = g[:, :, 0] / (n-1) * 2 - 1
    y2 = g[:, :, 1] / (m-1) * 2 - 1
    f = y1 ** 2 - 2 * y1 ** 2 * y2 ** 2 + y2 ** 2

    if opposite:
        a = (y2 - y1)/2 # slope of line through (-1, y1) and (1, y2).
        b = (y1 + y2)/2 # intercept -"-
        # S_-1^1 f(x, ax+b)dx
        # https://www.wolframalpha.com/input/?i=integrate+x%5E2-2x%5E2%28ax%2Bb%29%5E2%2B%28ax%2Bb%29%5E2+dx+from+x%3D-1+to+1
        f_integral = - 2 / 15 * (a ** 2 - 5 * (b ** 2 + 1))
        estimate = f_integral / 2 * (2 + np.abs(y2 - y1)) / 2 # multiplied by L1 length.
        # we didn't use it here, but the indefinite integral is:
        # integral(x^2 - 2 x^2 (a x + b)^2 + (a x + b)^2) dx = -2/5 a^2 x^5 - a b x^4 + (a x + b)^3/(3 a) - (2 b^2 x^3)/3 + x^3/3 + constant
    else:
        # now we switch to 0-1 space, and work with the line through (0, c) and (d, 0).
        c = (y1 + 1) / 2 # x distance from corner, in 0-1 space.
        d = (y2 + 1) / 2 # y distance from same corner, in 0-1 space.
        # https://www.wolframalpha.com/input/?i=integrate+%282x-1%29%5E2-2%282x-1%29%5E2%282y-1%29%5E2%2B%282y-1%29%5E2+dx+from+x%3D0+to+d+where+y%3D-%28c%2Fd%29+x%2Bc
        # f_integral = -16/15*c**2*d**3 + (8*c**2*d**2)/3 - (4*c**2*d)/3 + (8*c*d**3)/3 - (16*c*d**2)/3 + 2*c*d - (4*d**3)/3 + 2*d**2
        # estimate = f_integral / d * (c + d)
        # which trivially, but (for future reference) also according to
        # https://www.wolframalpha.com/input/?i=%28-16%2F15+c%5E2+d%5E3+%2B+%288+c%5E2+d%5E2%29%2F3+-+%284+c%5E2+d%29%2F3+%2B+%288+c+d%5E3%29%2F3+-+%2816+c+d%5E2%29%2F3+%2B+2+c+d+-+%284+d%5E3%29%2F3+%2B+2+d%5E2%29%28c%2Bd%29%2Fd
        # simplifies to
        estimate = 2/15 * (-8*c**2*d**2 + 20*c**2*d - 10*c**2 + 20*c*d**2 - 40*c*d + 15*c - 10*d**2 + 15*d) * (c + d)

    print("min", estimate.min(), "max", estimate.max(), "sum", f.mean())
    plt.imshow(estimate, vmin=0.0, vmax=0.7)
    plt.show()


def create_lagrangian(shape):
    g = mygrid((n-1, m-1)).astype(float) # reusing a code for cells that was created for crossings. (as in chess vs go)
    x = g[:, :, 0] / (n-1) * 2 - 1
    y = g[:, :, 1] / (m-1) * 2 - 1

    # the pictureframe L_infinity Langrangian.
    # lag = (np.maximum(np.abs(x), np.abs(y)) > 1 / 3).astype(int)

    # fitted polynomial
    lag = x * x + y * y - 2 * x * x * y * y
    return lag


def slices(line, shape):
    n, m = shape
    a, b, c = line
    ev = a * np.arange(n + 1)[:, np.newaxis] + b * np.arange(m + 1)[np.newaxis, :] + c
    evs = np.stack([ev[:-1, :-1], ev[1:, :-1], ev[:-1, 1:], ev[1:, 1:]], axis=-1)
    lows = np.min(evs, axis=2) < 0
    highs = np.max(evs, axis=2) > 0
    result = np.logical_and(lows, highs)
    return result


def vis_empirical(shape, lag, samples, opposite):
    n, m = shape
    N = samples
    print(lag.sum() / n / n)
    estimate = np.zeros((N, N))
    for i1, y1 in enumerate(np.linspace(-1, 1, N)):
        for i2, y2 in enumerate(np.linspace(-1, 1, N)):
            line = parametrized_line_on_opposite(shape, y1, y2, opposite=opposite)
            result = slices(line, shape).astype(int)
            estimate[i1, i2] = (result * lag).sum() / n
    plt.imshow(estimate, vmin=0.0, vmax=0.7)
    plt.show()


def compare_analytical_and_empirical_surfaces(shape):
    lag = create_lagrangian(shape)
    vis_quadratic(99, opposite=True)
    vis_empirical(shape, lag, samples=99, opposite=True)
    vis_quadratic(99, opposite=False)
    vis_empirical(shape, lag, samples=99, opposite=False)


compare_analytical_and_empirical_surfaces(shape) ; exit()


cache = "set-systems/%d-%d.npy" % (n, n)
slices = np.load(open(cache, "rb"))
print(slices.shape)

lag = create_lagrangian(shape)
ss = slices
ss = ss.reshape((len(ss), -1)) # flatten the nxm grids to n*m vectors
ss = ss.T

constraints = lag.flatten().dot(ss)
worst = constraints.max()
print(constraints.shape, worst)
lb = lag.sum() / worst
print("lb =", lb, ", that's", lb / n, "times n")
