import pytest
import numpy as np

from src.output_manager import OutputManager 
from tests.conftest import output_manager_fixture 

import tempfile
import os

def test_init():
    with tempfile.TemporaryDirectory() as tmpdirname:
        om = OutputManager(file_type = "bmp", output_dir = tmpdirname)
        assert om.file_type == "bmp"
    
def test_get_filename(output_manager_fixture):
    om = output_manager_fixture
    om._get_filename()
    assert om.filename == f"{om.output_dir}/img_0.{om.file_type}"

def test_save_image(output_manager_fixture):
    with tempfile.TemporaryDirectory() as tmpdirname:
        om = OutputManager(output_dir = tmpdirname) 
        image = np.random.uniform(0, 255, size = (100, 100, 3))
        for i in range(10):
            om.save(image)
            assert os.path.isfile(os.path.join(om.output_dir, om.filename))
