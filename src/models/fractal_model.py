import numpy as np
from dataclasses import dataclass
from fractions import Fraction
from itertools import permutations
from typing import Optional

DEFAULT_FSA: np.ndarray = np.array([[1, 2, 3, 4], 
                            [1, 2, 0, 4], 
                            [1, 2, 3, 0], 
                            [0, 2, 3, 4], 
                            [1, 0, 3, 4]])

from src.utils import pad_to_dense

class FractalModel:
    def __init__(self, generators: np.ndarray, special_fract: Fraction, FSA: Optional[np.ndarray] = DEFAULT_FSA): 
        self.generators: np.ndarray = generators
        self.special_fract: Fraction = special_fract
        self._compute_special_word()
        self._compute_fixed_points()
        self.FSA = FSA

    def _mobius_fixed_point(self, mat: np.ndarray):
        coeff = np.array([mat[1, 1], (mat[1, 1] - mat[0, 0]), -mat[0, 1]])
        return np.roots(coeff)

    def _compute_special_word(self):
        #See pp. 276
        num = 1
        c = self.special_fract.denominator
        special_word = []
        while True:
            if (num + self.special_fract.denominator > self.special_fract.numerator + self.special_fract.denominator):
                c = -self.special_fract.numerator 
                gen = 0
            elif (num - self.special_fract.numerator < 1):
                c = self.special_fract.denominator
                gen = 3

            num += c
            special_word.append(gen)
            if num == 1:
                break
        self.special_word = special_word[::-1]

    def _compute_fixed_points(self):
        fix_pts = [[] for i in range(4)]
        for spe_w in [self.special_word, [0, 1, 2, 3]]:
            for perm in set(permutations(spe_w)):
                word = self.generators[perm[0]]
                for p in perm[1:]:
                    word = np.matmul(word, self.generators[p])

                idx_gen = perm[-1]
                for w in self._mobius_fixed_point(word).flatten().tolist():
                    fix_pts[idx_gen].append(w)

        self.fixed_points = pad_to_dense(fix_pts)
