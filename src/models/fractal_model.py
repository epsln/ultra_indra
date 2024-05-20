import numpy as np
from dataclasses import dataclass
from fractions import Fraction


@dataclass
class FractalModel:
    generators: np.ndarray
    FSA: np.ndarray = np.array([[1, 2, 3, 4], 
                                [1, 2, 0, 4], 
                                [1, 2, 3, 0], 
                                [0, 2, 3, 4], 
                                [1, 0, 3, 4]])
    special_fract: Fraction
