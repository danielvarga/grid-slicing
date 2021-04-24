
# falsifying Gergo's pretty conjecture:
# python vis_monotone.py output/solution-mono.7.53850.npy
# python vis_monotone.py output/solution-mono.6.49575.npy

import numpy as np
import sys

input_filename, = sys.argv[1:]

ss = np.load(open(input_filename, "rb"))

print(ss.shape)


def pretty(s):
    return str(s.astype(int)).replace('1', '■').replace('0', '·').replace('[', ' ').replace(']', ' ').replace(' ', '')


agg = np.zeros_like(ss[0])
for r in ss:
    print("_____")
    print(pretty(r))
    agg += r

print("coverage:")
print(agg)

