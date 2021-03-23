import numpy as np
from sympy import *
import matplotlib.pyplot as plt

init_printing(use_unicode=True)


x, y, y1, y2, n = symbols('x y y1 y2 n')

xp = 2 * x - 1
yp = 2 * y - 1

f = xp * xp + yp * yp - 2 * xp * xp * yp * yp
# f = 0.22662088 + xp * xp * 1.3398939 + yp * yp * 1.3376968 - xp * xp * yp * yp * 2.3765702


double_integral = integrate(integrate(f, (x, 0, 1)), (y, 0, 1))
sum_on_grid = double_integral * n ** 2
print("sum on the grid is in the limit", sum_on_grid)


def opposite(y1, y2):
    # y1, y2 in (0, 1), representing the line connecting (0, y1) and (1, y2)
    a = y2 - y1
    b = y1

    line = a * x + b
    # -> equation for line through (0, y1) and (1, y2)

    f_on_line = f.subs(y, line)
    # -> now a function of x, y1, y2.

    integral = integrate(f_on_line, (x, 0, 1))
    # -> now only a function of y1 and y2.

    true_sum_on_line = integral * (n + n * abs(y1 - y2))
    sum_on_line = integral * (n + n * (y2 - y1)) # if we assume by symmetry that y2 >= y1, we get a polynomial
    # -> from now on we need to optimize on the ((0, 0) (0, 1) (1, 1)) triangle, not on the unit square.

    bound_for_line = sum_on_grid / sum_on_line
    # -> for each line, now parametrized by (y1, y2), we get a bound
    # for the set cover size, the true bound is the worst of these bounds.
    trend_for_line = bound_for_line / n
    # -> trend as in how much increasing n increases the cover size, based on this specific line.
    # this removes dependence on n, which was only added for clarity.
    trend_for_line = simplify(trend_for_line)

    inverse_trend_for_line = simplify(1 / trend_for_line)
    # -> this is easier to deal with, because it's a polynomial of (y1, y2) (if f is a polynomial)
    print("inverse_trend_for_line", inverse_trend_for_line)
    print("derivative", diff(inverse_trend_for_line, y1), diff(inverse_trend_for_line, y2))
    return inverse_trend_for_line

def neighboring(x1, y1):
    # x1, y1 in (0, 1), representing the line connecting (y1, 0) and (x1, 0)
    a = - y1 / x1
    b = y1

    line = a * x + b
    # -> equation for line

    f_on_line = f.subs(y, line)
    # -> now a function of x, x1, y1.

    integral = integrate(f_on_line, (x, 0, x1))
    # -> now only a function of x1 and y1.

    sum_on_line = integral / x1 * n * (x1 + y1)

    bound_for_line = sum_on_grid / sum_on_line
    trend_for_line = bound_for_line / n
    trend_for_line = simplify(trend_for_line)
    inverse_trend_for_line = simplify(1 / trend_for_line)
    # -> this is easier to deal with, because it's a polynomial of (x1, y1) (if f is a polynomial)
    return inverse_trend_for_line


fluff_text = "we have an alpha*n lower bound for the set size if we prove that this is smaller than 1/alpha everywhere on the"
do_opposite = True
if do_opposite:
    print("line between opposite sides:")
    inverse_trend_for_line = opposite(y1, y2)
    print("inverse_trend_for_line", inverse_trend_for_line)
    print(fluff_text, "((0, 0) (0, 1) (1, 1)) triangle.")
else:
    print("line between neighboring sides:")
    inverse_trend_for_line = neighboring(y1, y2)
    print("inverse_trend_for_line", inverse_trend_for_line)
    print(fluff_text, "01 square.")


inverse_trend_np = lambdify((y1, y2), inverse_trend_for_line, "numpy")

N = 50
y1g, y2g = np.meshgrid(np.linspace(0, 1, N), np.linspace(0, 1, N))
bound = inverse_trend_np(y1g, y2g)
if do_opposite:
    pass # bound[y1g > y2g] = 0 # the assumption fails for these pairs

plt.imshow(bound)
plt.show()
