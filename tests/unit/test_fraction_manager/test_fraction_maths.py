import pytest
import numpy as np
from fractions import Fraction

import src.fraction_manager.fraction_math as fm

def test_gcd():
    assert fm.gcd(12, 6) == 6
    assert fm.gcd(10, 52) == 2

def test_poly():
    assert fm.trace_poly(Fraction(0, 1), 2, 2, 2) == 2

def test_trace_equation():
    assert fm.trace_equation(Fraction(0, 1), 2) == -2-2j
