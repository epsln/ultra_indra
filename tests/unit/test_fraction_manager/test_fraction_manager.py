import pytest
import numpy as np
from fractions import Fraction

from tests.conftest import fraction_manager_fixture as fm
from src.fraction_manager import FractionManager 
from src.fraction_manager.fraction_manager import FractionManager

def test_init():
    with pytest.raises(ValueError):
        fm = FractionManager(root_iter = 1, root_epsilon = 1e-6)
    with pytest.raises(ValueError):
        fm = FractionManager(root_iter = 100, root_epsilon = 1e+6)
    with pytest.raises(ValueError):
        fm = FractionManager(root_iter = 100, root_epsilon = -1)

