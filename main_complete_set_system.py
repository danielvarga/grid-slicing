
# 1. connects all pairs of grid points.
# 2. collects all the points where the one of the above lines intersects left side of the board.
# 3. takes representative (midpoint) of each of the resulting intervals.
# 4. for each such representative r, take collect_star(r), a set of digital lines.
# 5. take union of these.
# 6. add vertical digital lines, cause they are still missing at the moment.

# collect_star(r): emit rays from r toward each grid point.
#   collect directions. take midpoint d of each interval of directions (circular arc).
#   for line across d and direction d, collect the digital line.

import numpy as np
import cvxpy as cp
from fractions import Fraction
import matplotlib.pyplot as plt
import sys

from pysat.formula import WCNF
from pysat.examples.rc2 import RC2
from pysat.solvers import Glucose3
from pysat.card import CardEnc


# returns an a/b Fraction. it answers this question:
# at what rational (a/b, 0) does the line between p1 and p2 intersect the x axis?
# not vectorized.
def intersect(p1, p2):
    x1, y1 = p1
    x2, y2 = p2
    dy = y2 - y1
    dx = x2 - x1
    a = x1 * dy - y1 * dx
    if dy == 0:
        return Fraction(0) # yeah i know nan != 0
    else:
        return Fraction(a, dy)


def test_intersect():
    n = 20
    import matplotlib.pyplot as plt
    for _ in range(5):
        ps = np.random.randint(n, size=(3, 2))
        x = float(intersect(ps[0], ps[1]))
        # x = pq[0] / pq[1]
        ps = ps.astype(float)
        ps[2, 0] = x
        ps[2, 1] = 0
        plt.scatter(ps[:, 0], ps[:, 1])
    plt.show()


# test_intersect() ; exit()


def make_grid(n):
    l = np.arange(n, dtype=int)
    g = np.stack(np.meshgrid(l, l), axis=-1)
    gf = g.reshape(-1, 2)
    return gf


# when you connect any pair of grid points with a line,
# what are the points where these lines intersect the x axis?
def find_crossings(n):
    gf = make_grid(n)
    crossings = set()
    for p1 in gf:
        for p2 in gf:
            x = intersect(p1, p2)
            crossings.add((x.numerator, x.denominator))

    crossings = sorted([Fraction(num, den) for num, den in crossings])
    return crossings


# after crossings slice up the x axis into intervals,
# we take an internal point from each interval.
def find_midpoints(n):
    crossings = find_crossings(n)
    crossings = [crossings[0] - 1] + crossings + [crossings[-1] + 1]
    mids = []
    for i in range(len(crossings) - 1):
        mid = (crossings[i] + crossings[i + 1]) / 2
        mids.append(mid)
    return mids


def collect_star(cx, cy, n):
    gf = make_grid(n)
    gf = gf.astype(object)
    gf[:, 0] -= cx
    gf[:, 1] -= cy
    zerodivs = gf[:, 0] == 0
    assert zerodivs.sum() == 0
    derivatives = gf[:, 1] / gf[:, 0]
    derivatives = sorted(derivatives)
    mids = []
    # we leave out the vertical line,
    # will have to put them back at the very end.
    for i in range(len(derivatives) - 1):
        mid = (derivatives[i] + derivatives[i + 1]) / 2
        mids.append(mid)

    # our lines are y = mid * (x-cx) + cy
    lines = [(mid, cy - mid * cx) for mid in mids]
    return lines


# returns an (n - 1) x (n - 1) shaped grid of 0-1 pixels,
# n is the number of points, like in Go,
# not the number of pixels, like in Chess.
def digital_line(line, n):
    # ax+by+c=0 is our intersecting line
    a, b, c = line
    # evaluate the left hand side on the n x n (Go) grid:
    ev = a * np.arange(n)[:, np.newaxis] + b * np.arange(n)[np.newaxis, :] + c
    evs = np.stack([ev[:-1, :-1], ev[1:, :-1], ev[:-1, 1:], ev[1:, 1:]], axis=-1)
    assert evs.shape == (n-1, n-1, 4) # this is now a Chess grid.
    lows = np.min(evs, axis=2) < 0 # did any of its 4 corners go below 0?
    highs = np.max(evs, axis=2) > 0 # did any of its 4 corners go above 0?
    result = np.logical_and(lows, highs) # both?
    return result


# digital_line((1, 1, 0.2), 10)


def vis_frac_line(line, **kwargs):
    p, q = line # y = p * x + q
    # For non-vertical lines, choose two x values
    x = np.array([-100, 100])
    # Calculate corresponding y values
    y = (p * x + q).astype(float)
    plt.plot(x, y, **kwargs)


def vis_abc_line(line, **kwargs):
    a, b, c = line
    if b != 0:
        # For non-vertical lines, choose two x values
        x = np.array([-100, 100])
        # Calculate corresponding y values
        y = (-a / b) * x - (c / b)
    else:
        # For a vertical line, x is constant
        x = np.array([-c / a, -c / a])
        # Choose y range (for example, from -10 to 10)
        y = np.array([-100, 100])
    plt.plot(x, y, **kwargs)


# for n in range(2, 25): print(f"{n}\t{len(find_midpoints(n))}")


def frac_to_abc(l):
    p, q = l # y = p*x + q line, p and q are Fractions.
    # we multiply by both denominators and reorder to 0 = ax+by+c :
    return (p.numerator * q.denominator, - p.denominator * q.denominator, p.denominator * q.numerator)


def star_test():

    n = 5

    frac_lines = collect_star(Fraction(3, 2), Fraction(10, 3), n)
    abc_lines = [frac_to_abc(l) for l in frac_lines]

    for i in range(n):
        vis_abc_line((-1, 0, i), c='lightblue')
        vis_abc_line((0, -1, i), c='lightblue')


    # for frac_line in frac_lines: vis_frac_line(frac_line)
    for abc_line in abc_lines: vis_abc_line(abc_line)

    plt.xlim(-0.5, 2*n-0.5)
    plt.ylim(-0.5, 2*n-0.5)
    plt.show()
    exit()


# star_test() ; exit()


# does not do unique
def symmetries(c):
    assert c.shape[1] == c.shape[2]
    m = c.shape[1]
    cs = []
    for cnt in range(4):
        c_prime = np.rot90(c, k=cnt, axes=(1, 2))
        cs.append(c_prime)
    cs = np.array(cs)
    cs = cs.reshape(-1, m, m)
    return cs


def collect_digital_lines(n):
    mids = find_midpoints(n)

    collection = []

    for mid in mids:
        # print(mid)
        frac_lines = collect_star(mid, Fraction(0), n)
        # for frac_line in frac_lines: print(frac_line)

        abc_lines = [frac_to_abc(l) for l in frac_lines]
        for abc_line in abc_lines:
            dl = digital_line(abc_line, n).astype(int)
            # print(dl)
            collection.append(dl)

    collection = np.array(collection)
    collection = symmetries(collection)

    collection = collection.reshape(len(collection), -1)
    collection = np.unique(collection, axis=0)
    collection = collection.reshape(len(collection), n-1, n-1)

    return collection


def minimal_cover(matrices):
    """
    Given a list of 0/1 matrices (each represented as a list of lists),
    returns the indices of a minimal subset of matrices whose element-wise
    logical OR produces an all-ones matrix.
    
    Args:
        matrices: List of matrices. Each matrix is a list of lists where each 
                  entry is either 0 or 1. All matrices should be of the same size.
    
    Returns:
        A list of indices (0-indexed) indicating which matrices are selected.
    
    Raises:
        ValueError: If any cell in the target matrix is not covered by any matrix.
    """
    # Assume all matrices have dimensions m x n:
    m = len(matrices[0])
    n = len(matrices[0][0])
    num_matrices = len(matrices)

    # Create a weighted CNF instance
    wcnf = WCNF()

    # -----------------------------
    # Hard constraints:
    # -----------------------------
    # For every cell (i, j) we add a clause saying that at least one matrix that has a 1
    # at that cell must be selected.
    for i in range(m):
        for j in range(n):
            clause = []
            # Enumerate matrices (using 1-indexed literals for PySAT)
            for k, M in enumerate(matrices, start=1):
                if M[i][j] == 1:
                    clause.append(k)
            # If no matrix covers cell (i, j), no solution exists.
            if not clause:
                raise ValueError(f"Cell ({i}, {j}) is not covered by any matrix.")
            wcnf.append(clause)  # Adding as a hard clause
    
    # -----------------------------
    # Soft constraints:
    # -----------------------------
    # We add a soft clause for each matrix variable:
    # The clause [-k] (where k is the literal corresponding to matrix k) is satisfied 
    # if matrix k is not chosen. If matrix k is chosen (i.e. x_k = True), then the clause is unsatisfied
    # and a penalty of 1 is incurred. Thus, the solver is encouraged to set x_k to False.
    for k in range(1, num_matrices + 1):
        wcnf.append([-k], weight=1)
    
    # -----------------------------
    # Solve with RC2 (a MaxSAT solver)
    # -----------------------------
    with RC2(wcnf) as rc2:
        solution = rc2.compute()
    
    # The solution is a list of literals. Positive literal k means the corresponding matrix is selected.
    # Convert these (which are 1-indexed) to 0-indexed list.
    selected = [k - 1 for k in solution if k > 0]
    return selected


def minimal_cover_iter(matrices):
    """
    Given a list of 0/1 matrices (each represented as a list of lists),
    returns the indices of a minimal subset of matrices whose element-wise
    logical OR produces an all-ones matrix.

    Args:
        matrices: List of matrices. Each matrix is a list of lists where each 
                  entry is either 0 or 1. All matrices should be the same size.

    Returns:
        A list of indices (0-indexed) of the selected matrices. Returns None if no solution exists.
    """
    m = len(matrices[0])       # number of rows
    n = len(matrices[0][0])    # number of columns
    num_matrices = len(matrices)
    
    # Precompute the coverage clauses for each cell.
    # Each clause is a list of variables (using 1-indexed IDs) that cover that cell.
    cell_clauses = []
    for i in range(m):
        for j in range(n):
            clause = []
            # Enumerate matrices (variables are 1-indexed)
            for idx, matrix in enumerate(matrices, start=1):
                if matrix[i][j] == 1:
                    clause.append(idx)
            if not clause:
                raise ValueError(f"Cell ({i}, {j}) is not covered by any matrix.")
            cell_clauses.append(clause)
    
    # Try different cardinality bounds from 1 to num_matrices.
    for k in range(1, num_matrices + 1):
        print("trying to find solution of size", k)
        solver = Glucose3()
        
        # Add the coverage clauses (these are hard constraints).
        for clause in cell_clauses:
            solver.add_clause(clause)
        
        # Add a cardinality constraint: At most k matrices are selected.
        # The decision variables are the matrix selection variables: 1, 2, ..., num_matrices.
        card = CardEnc.atmost(lits=list(range(1, num_matrices + 1)), bound=k, encoding=1)
        for clause in card.clauses:
            solver.add_clause(clause)
        
        # Try to solve the current SAT instance.
        if solver.solve():
            model = solver.get_model()
            # Extract selected matrix indices (convert 1-indexed to 0-indexed).
            selected = [lit - 1 for lit in model if 1 <= abs(lit) <= num_matrices and lit > 0]
            return selected
        solver.delete()

    # If no cover exists, return None.
    return None


n, = map(int, sys.argv[1:])


collection = collect_digital_lines(n)

print("number of digital lines", len(collection))
# for dl in collection:
#     print(dl)


k = len(collection) # no sampling
indices = np.random.choice(len(collection), size=k, replace=False)
collection = collection[indices]
print("sampled", k, "lines")


def minimal_cover_cvxpy(matrices):
    """
    Given a list of 0/1 matrices (each represented as a list of lists),
    returns the indices of a minimal subset of matrices whose element-wise
    logical OR produces an all-ones matrix.
    
    Args:
        matrices: List of matrices. Each matrix is a list of lists with entries 
                  0 or 1. All matrices should have the same dimensions.
    
    Returns:
        A list of indices (0-indexed) of the selected matrices. Returns None if
        no cover exists.
    """
    num_matrices = len(matrices)
    m = len(matrices[0])       # number of rows
    n = len(matrices[0][0])    # number of columns

    # Decision variables: one binary variable per matrix.
    x = cp.Variable(num_matrices, boolean=True)

    # Build coverage constraints: for every cell (i, j) at least one matrix that covers the cell must be selected.
    constraints = []
    for i in range(m):
        for j in range(n):
            # We sum the x[k] of those matrices that have a 1 at cell (i, j).
            cell_cover = []
            for k, matrix in enumerate(matrices):
                if matrix[i][j] == 1:
                    cell_cover.append(x[k])
            # If no matrix covers this cell, the problem is unsolvable.
            if len(cell_cover) == 0:
                raise ValueError(f"Cell ({i}, {j}) is not covered by any matrix.")
            constraints.append(sum(cell_cover) >= 1)

    # Objective: minimize the total number of selected matrices.
    objective = cp.Minimize(cp.sum(x))
    
    # Define and solve the problem.
    prob = cp.Problem(objective, constraints)
    # Choose a mixed-integer solver (ensure one is installed; GLPK_MI is often a good free option)
    prob.solve(solver=cp.GUROBI)

    if prob.status in ['optimal', 'optimal_inaccurate']:
        # Retrieve the solution and round if necessary.
        sol = x.value
        # Convert the solution to binary (0 or 1)
        sol_binary = [int(round(v)) for v in sol]
        selected = [i for i, v in enumerate(sol_binary) if v == 1]
        return selected
    else:
        return None


selected_idx = minimal_cover_cvxpy(collection)
# selected_idx = minimal_cover(collection)
# selected_idx = minimal_cover_iter(collection)
print(">>>", selected_idx)
print(collection[selected_idx, :, :])

print("cover", collection[selected_idx, :, :].sum(axis=0))
print("grid size", n - 1, "solution size", len(selected_idx))
