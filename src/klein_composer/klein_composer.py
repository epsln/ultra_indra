from klein_compute.tree_explorer import tree_explorer 
from src.klein_composer.utils import pad_to_dense

import numpy as np
import threading
from fractions import Fraction
from itertools import permutations
import logging

_logger = logging.getLogger(__name__)
class KleinComposer():
    def __init__(self):
        self.gen = np.random.uniform(-1, 1, size = (4, 2, 2)) + 1.j * np.random.uniform(-1, 1, size = (4, 2, 2))
        self.fsa = np.array([[1, 2, 3, 4], 
                             [1, 2, 0, 4], 
                             [1, 2, 3, 0], 
                             [0, 2, 3, 4], 
                             [1, 0, 3, 4]])
        self.num_threads = 4
        self.special_fract = Fraction(0, 1) 
        #self.word_length = special_word.p + special_word.q
        #self.special_word = self.compute_special_word()
        self.word_length = 0
        self.fix_pts = [[] for i in range(4)]
        #self.fix_pts = self.compute_fix_pts(self.special_word)
        self.special_word = [0]
        self.fix_pts = self.compute_fix_pts(self.special_word) #special word abAB

        self.tree_exp = tree_explorer(0, 0, 3, self.gen, self.fsa, self.fix_pts)

    def mobius_fixed_point(self, mat):
        coeff = np.array([mat[1, 1], (mat[1, 1] - mat[0, 0]), -mat[0, 1]])
        return np.roots(coeff)

    def compute_fix_pts(self, special_words):
        fix_pts = [[] for i in range(4)]
        for spe_w in [special_words, [0, 1, 2, 3]]: 
            for perm in set(permutations(spe_w)): 
                word = self.gen[perm[0]]
                for p in perm[1:]:
                    word = np.matmul(word, self.gen[p])

                idx_gen =  perm[-1]
                for w in self.mobius_fixed_point(word).flatten().tolist():
                    fix_pts[idx_gen].append(w)
            
        return pad_to_dense(fix_pts)

    def compute_start_points(self):
        return self.tree_exp.compute_leaf()

    def compute_thread(self, depth):
        return self.tree_exp.compute_leaf()

    def compute_thread(self, depth):
        start_points = self.compute_start_points()
        for n in start_points:
            if n_threads == self.num_threads:
                #Check if one is done
                continue
            thread = threading.Thread(target = compute_leaf, args = ())

