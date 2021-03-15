import numpy as np
import sys

input_filename, = sys.argv[1:]

ss = np.load(open(input_filename, "rb"))


# this is just a crude heuristic, can often lead to mistakes.
def reconstruct(line_set):
    x, y = line_set.nonzero()
    # we add 0.5 because we fit on the center of squares
    slope, intercept = np.polyfit(x + 0.5, y + 0.5, 1)
    return slope, intercept


def visualize_solution(ss, output_filename):
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
    for line_set in ss:
        '''
        squares_x, squares_y = line_set.nonzero()
        for x, y in zip(squares_x, squares_y):
            draw.ellipse((x * d, y * d, (x + 1) * d, (y + 1)* d), fill='green')
        '''

        slope, intercept = reconstruct(line_set)
        y_last = None
        for x in range(N):
            y = d * (slope * x / d + intercept)
            y_last = y
            if y_last is not None:
                draw.line((x - 1, y_last, x, y), fill='red')

    im.save(output_filename)


output_filename = input_filename.replace('.npy', '.png')

lines = np.array([reconstruct(line_set) for line_set in ss])

n, m = ss.shape[1:]
assert n == m

# this is very specific to configurations of n=10 solutions, see ./solutions-batch/10/
# after normalization, the 2-batch goes WNW-to-ESE (slopes.max() > 1), and the 7-batch goes somewhere near NE-to-SW.
normalize_slopes = True
if normalize_slopes:
    slopes = lines[:, 0]
    # assert (slopes > 0).sum() in (2, 7)
    if (slopes > 0).sum() > n / 2:
        # the majority should be negative
        slopes = - slopes
        ss = ss[:, :, ::-1]

    # normal form, corresponding to a matrix transpose, turning NNW into WNW:
    if slopes.max() > 1.0:
        slopes = 1.0 / slopes
        ss = ss.transpose((0, 2, 1))
    perm = np.argsort(slopes)
    slopes = slopes[perm]
    lines = lines[perm]
    ss = ss[perm]
else:
    slopes = lines[:, 0]


just_dump_slopes = False
if just_dump_slopes:
    print("\t".join(map(lambda x: "%1.4f" % x, slopes)))
    exit()


print("line equations:")
# note: lines are not normalized, because they were not mirrored and transposed unlike slopes and ss.
print(lines)

visualize_solution(ss, output_filename)

subset = [i for i in range(len(slopes)) if slopes[i] > 0]

complement = list(set(range(len(slopes))).difference(subset))
if len(subset) > len(complement):
    subset, complement = complement, subset


def pp(l):
    return str(list(l)).replace('[]', '.').replace('[', '').replace(']', '').replace(' ', '')

def compact_print(ss):
    grid = ss.transpose((1, 2, 0))
    n, m, s = grid.shape
    for i in range(n):
        for j in range(m):
            print(pp(grid[j, i].nonzero()[0]), "\t", end='')
        print()

ss1 = ss.copy()
ss1[complement, :, :] = 0
compact_print(ss1)
print()
ss2 = ss.copy()
ss2[subset, :, :] = 0
compact_print(ss2)
