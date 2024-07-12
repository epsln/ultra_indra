from src.tree_explorer.tree_explorer import compute_tree

import numpy as np
from multiprocessing import Pool
import logging

_logger = logging.getLogger(__name__)


class KleinComposer:
    def __init__(self, fractal_model, compute_model, output_model):
        self.gen = fractal_model.generators
        self.fsa = fractal_model.FSA
        self.fsa = np.array(
            [[1, 2, 3, 4], [1, 2, 0, 4], [1, 2, 3, 0], [0, 2, 3, 4], [1, 0, 3, 4]]
        )
        self.num_threads = compute_model.num_threads
        self.max_depth = compute_model.max_depth
        self.special_fract = fractal_model.special_fract
        self.word_length = 0
        self.fix_pts = fractal_model.fixed_points
        self.start_depth = self.compute_start_depth(self.num_threads)

        self.cm = compute_model
        self.fm = fractal_model
        self.om = output_model

        # TODO: Rename threads into workers
        self.pool = Pool(self.cm.num_threads)

    @staticmethod
    def compute_start_depth(num_threads):
        if num_threads <= 4:
            return 1
        else:
            return np.floor(np.log(num_threads - 4) / np.log(3))

    def compute_start_points(self):
        # TODO: Use the cython function
        # This is a bodge in the meantime
        output = []
        for i, g in enumerate(self.fm.generators):
            output.append((i, i + 1, g))

        return output

    def compute_thread(self):
        #TODO: Find some way to clean up argument passing
        start_elements = self.compute_start_points()
        arguments = [
            (
                e[0],
                e[1],
                e[2],
                self.cm.max_depth,
                self.cm.epsilon,
                self.fm.generators,
                self.fm.FSA,
                self.fm.fixed_points,
                self.fm.fixed_points_shape,
                self.om.image_dim,
                self.om.z_min,
                self.om.z_max 
            )
            for e in start_elements
        ]

        with self.pool as p:
            output = p.starmap(compute_tree, arguments)

        image = np.zeros(self.om.image_dim)
        for o in output:
            image = np.add(image, o)

        return image 
