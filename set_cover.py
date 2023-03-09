import sys
import numpy as np
import itertools

import gurobipy as gp
from gurobipy import GRB



def pretty(s):
    return str(s.astype(int)).replace('1', '■').replace('0', '·').replace('[', ' ').replace(']', ' ').replace(' ', '')



# shaped (2**n, n)
def powerset(n):
    combinations = itertools.product([0, 1], repeat=n)
    return np.array(list(combinations))


def all_jumps_from_zero(n):
    a = powerset(n)
    jumps = np.cumsum(a, axis=1)
    jumps = np.insert(jumps, 0, 0, axis=1)
    return jumps


def all_jumps(n):
    all_jumps = []
    zero_jump_list = all_jumps_from_zero(n)
    for jumps in zero_jump_list:
        last = jumps[-1]
        for delta in range(n - last):
            all_jumps.append(jumps + delta)
    return np.array(all_jumps)


def parametrized_monotone(n, jumps):
    assert len(jumps) == n + 1
    assert min(jumps) >= 0
    assert max(jumps) <= n - 1
    result = np.zeros((n, n), dtype=np.uint8)
    y_prev = None
    for x, y in enumerate(jumps): # x addresses a row, y a column!
        if x > 0:
            result[x - 1, y_prev : y + 1] = 1
        y_prev = y
    return result


def low_slope_gentle_monotones(n):
    jump_list = all_jumps(n)
    results = [parametrized_monotone(n, jumps) for jumps in jump_list]
    return np.array(results)


def all_gentle_monotones(n):
    r = low_slope_gentle_monotones(n)
    print("low slope", r.shape)
    r = np.concatenate([r, r[:, :, ::-1]])
    r = np.concatenate([r, r[:, ::-1, :]]) # this is not really needed
    r = np.concatenate([r, np.transpose(r, (0, 2, 1))])
    r = np.unique(r, axis=0)
    print("all", r.shape)
    return r


n = int(sys.argv[1])

set_system = all_gentle_monotones(n)


'''
a = np.load(sys.argv[1])
'''
set_system_size, n, m = set_system.shape
assert n == m
set_system = set_system.reshape((set_system_size, n * n))

try:
    m = gp.Model("set_cover")

    x = m.addVars(len(set_system), vtype=GRB.BINARY, name="is_in")

    # Set objective
    m.setObjective(gp.quicksum(x), GRB.MINIMIZE)

    m.addConstrs((gp.quicksum(set_system[j, i] * x[j] for j in range(set_system_size)) >= 1 for i in range(n * n)))

    # Optimize model
    m.optimize()

    '''
    for v in m.getVars():
        print('%s %g' % (v.VarName, v.X))
    '''

    scs = m.ObjVal
    print('optimal set cover size: %g' % scs)
    if scs < n - 1:
        print("SURPRISE!!!")

except gp.GurobiError as e:
    print('Error code ' + str(e.errno) + ': ' + str(e))

except AttributeError:
    print('Encountered an attribute error')
