import pytest

import numpy as np
from src.klein_composer import KleinComposer
from tests.conftest import fractal_model_fixture, compute_model_fixture, klein_composer_fixture, output_manager_fixture


def test_full_run(klein_composer_fixture, output_manager_fixture):
    p = klein_composer_fixture.compute_thread()
    output_manager_fixture.save(p)
