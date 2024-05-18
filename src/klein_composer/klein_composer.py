from klein_compute.tree_explorer import tree_explorer 

import numpy as np
import threading
from fractions import Fraction

class KleinComposer():
    def __init__(self):
        self.gen = np.random.uniform(-1, 1, size = (4, 2, 2)) + 1.j * np.random.uniform(-1, 1, size = (4, 2, 2))
        self.fsa = np.array([[1, 2, 3, 4], 
                             [1, 2, 0, 4], 
                             [1, 2, 3, 0], 
                             [0, 2, 3, 4], 
                             [1, 0, 3, 4]])
        self.num_threads = 4
        self.special_word = Fraction(0, 1) 
        self.tree_exp = tree_explorer(0, 0, 3, self.gen, self.fsa)

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

