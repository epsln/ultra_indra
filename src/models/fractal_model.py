import numpy as np
from dataclasses import dataclass
from fractions import Fraction


@dataclass
class FractalModel:
    generators: np.ndarray
    FSA: np.ndarray
    special_fract: Fraction
