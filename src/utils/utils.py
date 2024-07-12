import numpy as np


def pad_to_dense(M):
    """Appends the minimal required amount of zeroes at the end of each
    array in the jagged array `M`, such that `M` looses its jagedness."""

    maxlen = max(len(r) for r in M)

    Z = np.zeros((len(M), maxlen), dtype=complex)
    for enu, row in enumerate(M):
        Z[enu, : len(row)] += row
    return Z


def cyclic_permutation(a):
    n = len(a)
    b = [[a[i - j] for i in range(n)] for j in range(n)]
    return b
