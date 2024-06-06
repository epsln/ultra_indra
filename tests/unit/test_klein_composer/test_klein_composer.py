import pytest

import numpy as np
from src.klein_composer import KleinComposer
from tests.conftest import fractal_model_fixture, compute_model_fixture, klein_composer_fixture


def test_init(fractal_model_fixture, compute_model_fixture):
    kc = KleinComposer(fractal_model_fixture, compute_model_fixture)
    assert kc.num_threads == 4
    assert kc.word_length == 0
    assert kc.gen.shape == (4, 2, 2)
    assert kc.fsa.shape == (5, 4)
    
def test_compute_start_depth():
    assert KleinComposer.compute_start_depth(1) == 1
    assert KleinComposer.compute_start_depth(4) == 1
    assert KleinComposer.compute_start_depth(14) == 2

def test_compute_start_pts(klein_composer_fixture):
    start_pts, start_state, start_tags = klein_composer_fixture.compute_start_points() 
    for g in klein_composer_fixture.gen:
        assert g in start_pts

def test_compute_thread(klein_composer_fixture):
    kc = klein_composer_fixture 
    kc.compute_thread()
