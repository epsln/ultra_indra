from src.models import FractalModel, ComputeModel, OutputModel
from src.klein_composer import KleinComposer 
from src.output_manager import OutputManager 
import numpy as np
from fractions import Fraction
import pytest
import tempfile

@pytest.fixture
def fractal_model_fixture():
    return FractalModel(generators = np.random.uniform(-1, 1, (4, 2, 2)) + 1.j * np.random.uniform(-1, 1, (4, 2, 2)),
                        FSA = np.array([[1, 2, 3, 4], 
                                        [1, 2, 0, 4], 
                                        [1, 2, 3, 0], 
                                        [0, 2, 3, 4], 
                                        [1, 0, 3, 4]]),
                        special_fract = Fraction(0, 1)
                        )

@pytest.fixture
def compute_model_fixture():
    return ComputeModel(max_depth = 4,
                        num_threads = 4 
                        )

@pytest.fixture
def klein_composer_fixture(fractal_model_fixture, compute_model_fixture):
    return KleinComposer(fractal_model_fixture, compute_model_fixture)

@pytest.fixture
def output_model_fixture():
    with tempfile.TemporaryDirectory() as tmpdirname:
        return OutputModel(output_directory = tmpdirname)

@pytest.fixture
def output_manager_fixture(output_model_fixture):
        return OutputManager(output_model_fixture)
