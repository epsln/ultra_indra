import pytest
import numpy as np

from src.output_manager import OutputManager 
from tests.conftest import output_manager_fixture 

import tempfile
import os

def test_get_filename(output_manager_fixture):
    om = output_manager_fixture
    om._get_filename()
    assert om.filename == f"{om.output_dir}/img_0.{om.file_type}"

def test_save_image(output_manager_fixture):
    with tempfile.TemporaryDirectory() as tmpdirname:
        om = output_manager_fixture 
        for i in range(10):
            image =  np.random.uniform(size = (1080, 1080))
            om.save(image)
            assert os.path.isfile(os.path.join(om.output_dir, om.filename))
