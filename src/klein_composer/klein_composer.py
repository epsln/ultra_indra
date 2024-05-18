from klein_compute.tree_explorer import tree_explorer 
from src.klein_composer.utils import pad_to_dense
from src.klein_composer.utils import compute_fix_pts 

import numpy as np
import threading
import logging

_logger = logging.getLogger(__name__)
class KleinComposer():
    def __init__(self, fractal_scheme, compute_scheme):
        self.gen = fractal_scheme.generators 
        self.fsa = fractal_scheme.FSA
        self.fsa = np.array([[1, 2, 3, 4], 
                             [1, 2, 0, 4], 
                             [1, 2, 3, 0], 
                             [0, 2, 3, 4], 
                             [1, 0, 3, 4]])
        self.num_threads = compute_scheme.num_threads
        self.max_depth = compute_scheme.max_depth
        self.special_fract = fractal_scheme.special_fract 
        #self.word_length = special_word.p + special_word.q
        #self.special_word = self.compute_special_word()
        self.word_length = 0
        #self.fix_pts = self.compute_fix_pts(self.special_word)
        self.special_word = [0]
        self.fix_pts = compute_fix_pts(self.gen, self.special_word) #special word abAB
        self.fix_pts = pad_to_dense(self.fix_pts)

        self.tree_exp = tree_explorer(0, 0, 3, self.gen, self.fsa, self.fix_pts)

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

