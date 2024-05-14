from klein_compute.tree_explorer import tree_explorer 
import numpy as np

class KleinComposer():
    def __init__(self):
        gen = np.random.uniform(-1, 1, 2 * 2 * 4) + 1.j * np.random.uniform(-1, 1, 2 * 2 * 4)
        fsa = np.ndarray([0, 1, 2, 3, 2, 3, 0, 5, 2, 3, 4, 0, 0, 3, 4, 5, 2, 0, 4, 5])
        self.tree_exp = tree_explorer(0, 0, 2, gen, fsa)

    def compute_start_points():
        self.tree_exp.compute_leaf()
