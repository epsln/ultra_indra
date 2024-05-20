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

def test_compute_start_pts(klein_composer_fixture):
    start_pts = klein_composer_fixture.compute_start_points() 
    assert start_pts == klein_composer_fixture.gen
