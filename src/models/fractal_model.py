from src.utils import pad_to_dense

import numpy as np
from fractions import Fraction
from itertools import permutations
from typing import Optional

DEFAULT_FSA: np.ndarray = np.array(
    [[1, 2, 3, 4], [1, 2, 0, 4], [1, 2, 3, 0], [0, 2, 3, 4], [1, 0, 3, 4]]
)


class FractalModel:
    def __init__(
        self,
        generators: np.ndarray,
        special_fract: Fraction = Fraction(1, 2),
        FSA: Optional[np.ndarray] = DEFAULT_FSA,
    ):
        self.generators: np.ndarray = generators
        self.special_fract: Fraction = special_fract
        self._compute_special_word()
        self._compute_fixed_points()
        self.FSA = FSA

    def _mobius_fixed_point(self, mat: np.ndarray):
        coeff = np.array([mat[1, 0], (mat[1, 1] - mat[0, 0]), -mat[0, 1]])
        return np.roots(coeff)

    def _compute_special_word(self):
        # See pp. 276
        num = 1
        c = self.special_fract.denominator
        special_word = []
        while True:
            if (
                num + self.special_fract.denominator
                > self.special_fract.numerator + self.special_fract.denominator
            ):
                c = -self.special_fract.numerator
                gen = 0
            elif num - self.special_fract.numerator < 1:
                c = self.special_fract.denominator
                gen = 3

            num += c
            special_word.append(gen)
            if num == 1:
                break
        self.special_word = special_word[::-1]

    def _compute_fixed_points(self):
        fix_pts = [[] for i in range(4)]
        cleaned_spe_w = []
        for spe_w in [self.special_word, [0, 1, 2, 3]]:
            for perm in set(permutations(spe_w)):
                valid = True
                for curr_e, next_e in zip(perm, perm[1:]):
                    if (curr_e + 2) % 4 == next_e:
                        valid = False
                        break
                if valid: 
                    cleaned_spe_w.append(perm)
        
        for perm in cleaned_spe_w:
            word     = self.generators[perm[0]]
            word_inv = self.generators[(perm[0] + 2) % 4]
            for p in perm[1:]:
                word     = np.matmul(word, self.generators[p])
                word_inv = np.matmul(word_inv, self.generators[(p + 2) % 4])

            idx_gen = perm[-1]
            for w in self._mobius_fixed_point(word).flatten().tolist():
                fix_pts[idx_gen].append(w)

            for w in self._mobius_fixed_point(word_inv).flatten().tolist():
                fix_pts[idx_gen].append(w)

        self.fixed_points = pad_to_dense(fix_pts)
