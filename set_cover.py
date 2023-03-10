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

    # this doubling is apparently enough, no quadrupling is needed, but
    # all the other ways of doubling (mirror vertically, mirror horizontally, transpose) are not enough
    # to get an n-1 sized cover.
    # r = np.concatenate([r, np.transpose(r[:, :, ::-1], (0, 2, 1))])

    r = np.unique(r, axis=0)
    print("all", r.shape)
    return r


def solve_dual(set_system, already_covered):
    set_system_size, n, m = set_system.shape
    assert n == m
    set_system = set_system.reshape((set_system_size, n * n))
    assert already_covered.shape == (n, n)

    with gp.Env(empty=True) as env:
        # env.setParam('OutputFlag', 0)
        env.start()

        model = gp.Model("set_cover_dual", env=env)

        y = model.addVars(n * n, lb=0, ub=1, vtype=gp.GRB.CONTINUOUS, name="square")

        # Set objective
        model.setObjective(gp.quicksum(y), GRB.MAXIMIZE)

        model.addConstrs((gp.quicksum(set_system[j, i] * y[i] for i in range(n * n) if already_covered.flatten()[i] == 0) <= 1 for j in range(set_system_size)))

        # Optimize model
        model.optimize()

        y_pretty = np.array([v.X for v in model.getVars()]).reshape((n, n))
        y_pretty *= (1 - already_covered)
        scs = y_pretty.sum()

        if True:
            print(y_pretty)
            print('optimal weight: %g' % scs)
        return scs



def solve(set_system, already_covered):
    set_system_size, n, m = set_system.shape
    assert n == m
    set_system = set_system.reshape((set_system_size, n * n))
    assert already_covered.shape == (n, n)

    with gp.Env(empty=True) as env:
        # env.setParam('OutputFlag', 0)
        env.start()

        model = gp.Model("set_cover", env=env)

        x = model.addVars(len(set_system), vtype=GRB.BINARY, name="is_in")

        # Set objective
        model.setObjective(gp.quicksum(x), GRB.MINIMIZE)

        model.addConstrs((gp.quicksum(set_system[j, i] * x[j] for j in range(set_system_size)) >= 1 for i in range(n * n) if already_covered.flatten()[i] == 0))

        # Optimize model
        model.optimize()

        if True:
            for i, v in enumerate(model.getVars()):
                if v.X == 1:
                    print("====")
                    print(pretty(set_system[i].reshape((n, n))))

            scs = model.ObjVal
            print('optimal set cover size: %g' % scs)
        return model.ObjVal


def try_starters(n):
    pl = low_slope_gentle_monotones(n) # pl as in positive low slope
    nh = np.transpose(pl[:, :, ::-1], (0, 2, 1)) # nh as in negative high slope


    for i, already_covered in enumerate(nh):
        cover_size = solve(set_system=pl, already_covered=already_covered)
        if cover_size == n - 2:
            print(f"{i} can be finished to an optimal solution")
            print(pretty(already_covered))
        else:
            assert cover_size == n - 1
            # print(f"{i} unfinishable")


def main():
    n = int(sys.argv[1])

    already_covered = np.zeros((n, n))
    already_covered[n-2:, 0] = 1 ; already_covered[:2, n-1] = 1
    # already_covered = parametrized_monotone(n, [0]+list(range(n))) ; already_covered = already_covered.T[:, ::-1]
    print(pretty(already_covered))
    pl = low_slope_gentle_monotones(n) # pl as in positive low slope
    plph = np.concatenate([pl, np.transpose(pl, (0, 2, 1))]) # all positive slope lines

    cover_size = solve_dual(set_system=plph, already_covered=already_covered)
    print(cover_size, "supposedly", n-2)




if __name__ == "__main__":
    main() ; exit()
    # try_starters(n) ; exit()
