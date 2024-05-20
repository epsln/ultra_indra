import pytest

import numpy as np
from src.klein_composer import KleinComposer
from src.klein_composer.utils import mobius_fixed_point 
from tests.conftest import fractal_model_fixture, compute_model_fixture, klein_composer_fixture


def test_init(fractal_model_fixture, compute_model_fixture):
    kc = KleinComposer(fractal_model_fixture, compute_model_fixture)
    assert kc.num_threads == 4
    assert kc.word_length == 0
    assert kc.gen.shape == (4, 2, 2)
    assert kc.fsa.shape == (5, 4)

def test_fix_pts(fractal_model_fixture, compute_model_fixture):
    kc = KleinComposer(fractal_model_fixture, compute_model_fixture)
    
    test_word = np.matmul(kc.gen[0], kc.gen[1]) 
    test_word = np.matmul(test_word, kc.gen[2]) 
    test_word = np.matmul(test_word, kc.gen[3]) 
    assert mobius_fixed_point(test_word)[0] in kc.fix_pts  

def test_compute_start_pts(klein_composer_fixture):
    start_pts = klein_composer_fixture.compute_start_points() 
