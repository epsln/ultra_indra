import pytest
import numpy as np
from fractions import Fraction

from tests.conftest import fraction_manager_fixture as fm
from src.fraction_manager import FractionManager 
from src.fraction_manager.fraction_manager import FractionManager
from src.fraction_manager.fraction_math import trace_equation 

def test_init():
    with pytest.raises(ValueError):
        fm = FractionManager(fractions = [Fraction(0, 1)], root_iter = 1, root_epsilon = 1e-6)
    with pytest.raises(ValueError):
        fm = FractionManager(fractions = [Fraction(0, 1)], root_iter = 100, root_epsilon = 1e+6)
    with pytest.raises(ValueError):
        fm = FractionManager(fractions = [Fraction(0, 1)], root_iter = 100, root_epsilon = -1)

def test_init_from_farey():
    f = FractionManager.from_farey_sequence(max_denum = 5)
    assert len(f.fractions) == 10
    f = FractionManager.from_farey_sequence(max_denum = 10)
    assert len(f.fractions) == 32 
    
def test_generate(fm):
   fractal_model = fm.generate()

def test_trace_solver(fm):
   ta = fm.trace_solver(Fraction(1, 2), 2j)
   assert np.isclose(trace_equation(Fraction(1, 2), ta), 0)
