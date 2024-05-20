import pytest
import numpy as np
from fractions import Fraction

from src.models import FractalModel

def test_mobius_on_fp(fractal_model_fixture):
    a = 2 + 1j
    b = 3
    c = 1 + 2j
    d = 1
    arr = np.array([[a, b], [c, d]]) 
    expected = np.roots([c, d - a, -b])
    fp = fractal_model_fixture._mobius_fixed_point(arr)
    assert expected.all() == fp.all()

def test_compute_special_word(fractal_model_fixture):
    fractal_model_fixture.special_fract = Fraction(3, 5)
    expected = [0, 0, 3, 0, 0, 3, 0, 3]
    fractal_model_fixture._compute_special_word()
    assert expected == fractal_model_fixture.special_word
    fractal_model_fixture.special_fract = Fraction(2, 5)
    expected = [0, 0, 0, 3, 0, 0, 3]
    fractal_model_fixture._compute_special_word()
    assert expected == fractal_model_fixture.special_word

def test_compute_fixed_points(fractal_model_fixture):
    fm = fractal_model_fixture 
    test_word = np.matmul(fm.generators[0], fm.generators[1]) 
    test_word = np.matmul(test_word, fm.generators[2]) 
    test_word = np.matmul(test_word, fm.generators[3]) 
    fm._compute_fixed_points() 
    assert fm._mobius_fixed_point(test_word)[0] in fm.fixed_points 
