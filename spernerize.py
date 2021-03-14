import numpy as np
import sys

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


n = int(sys.argv[1])
set_system = np.load("%d-%d.npy" % (n, n))
kept = spernerize(set_system)
print("%d lines kept out of %d" % (len(kept), len(set_system)))
np.save(open("%d-%d.sperner.npy" % (n, n), "wb"), kept.astype(np.uint8))
