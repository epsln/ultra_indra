from klein_compute.tree_explorer import tree_explorer

import numpy as np
from multiprocessing import Pool
import logging

_logger = logging.getLogger(__name__)


class KleinComposer:
    def __init__(self, fractal_model, compute_model):
        self.gen = fractal_model.generators
        self.fsa = fractal_model.FSA
        self.fsa = np.array(
            [[1, 2, 3, 4], [1, 2, 0, 4], [1, 2, 3, 0], [0, 2, 3, 4], [1, 0, 3, 4]]
        )
        self.num_threads = compute_model.num_threads
        self.max_depth = compute_model.max_depth
        self.special_fract = fractal_model.special_fract
        self.word_length = 0
        self.special_word = [0]
        self.fix_pts = fractal_model.fixed_points
        self.start_depth = self.compute_start_depth(self.num_threads)

        self.cm = compute_model
        self.fm = fractal_model 

        #TODO: Rename threads into workers
        self.pool = Pool(self.cm.num_threads)


    @staticmethod
    def compute_start_depth(num_threads):
        if num_threads <= 4:
            return 1
        else:
            return np.floor(np.log(num_threads - 4)/np.log(3))

    def compute_start_points(self):
        tree_exp = tree_explorer(
            self.start_depth, self.cm.epsilon, self.fm.generators, self.fm.FSA, self.fm.fixed_points
        )
        return tree_exp.compute_tree()

    def compute_thread(self):
        tree_explorator = tree_explorer(
            self.max_depth, self.cm.epsilon, self.fm.generators, self.fm.FSA, self.fm.fixed_points
        )
        start_points, start_states, start_tags = self.compute_start_points()
        arguments = [(st, ss, self.start_depth, sp) for sp, ss, st in zip(start_points, start_states, start_tags)]
            
        with self.pool as p:
            output = p.starmap(tree_explorator.compute_leaf, arguments)
            
