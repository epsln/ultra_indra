import pytest

import numpy as np
from src.klein_composer import KleinComposer
from tests.conftest import fractal_model_fixture, compute_model_fixture, klein_composer_fixture


def test_compute_start_depth():
    assert KleinComposer.compute_start_depth(1) == 1
    assert KleinComposer.compute_start_depth(4) == 1
    assert KleinComposer.compute_start_depth(14) == 2

def test_compute_start_pts(klein_composer_fixture):
    start_element = klein_composer_fixture.compute_start_points() 

def test_compute_thread(klein_composer_fixture):
    kc = klein_composer_fixture 
    kc.compute_thread()
