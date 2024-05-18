import numpy as np
from itertools import permutations


def mobius_fixed_point(mat):
    coeff = np.array([mat[1, 1], (mat[1, 1] - mat[0, 0]), -mat[0, 1]])
    return np.roots(coeff)


def compute_special_word(fraction):
    pass


def compute_fix_pts(generators, special_words):
    fix_pts = [[] for i in range(4)]
    for spe_w in [special_words, [0, 1, 2, 3]]:
        for perm in set(permutations(spe_w)):
            word = generators[perm[0]]
            for p in perm[1:]:
                word = np.matmul(word, generators[p])

            idx_gen = perm[-1]
            for w in mobius_fixed_point(word).flatten().tolist():
                fix_pts[idx_gen].append(w)

    return fix_pts
