from src.klein_composer import KleinComposer
from src.models import FractalModel, ComputeModel, OutputModel
from src.output_manager import OutputManager
from src.recipe_manager import RecipeManager 
from src.fraction_manager import FractionManager 

from fractions import Fraction
import numpy as np

import logging
import os

if __name__ == "__main__":
    logging.basicConfig(
                level=os.environ.get('LOGLEVEL', 'INFO').upper()
                )
    om = OutputModel(file_type="jpg", output_directory="output")
    output_manager = OutputManager(om)
    fract_manager = FractionManager.from_farey_sequence(5)
    cm = ComputeModel(max_depth=40, epsilon = 0.0001, num_threads=4)

    for fm in fract_manager.generate():

        kc = KleinComposer(fm, cm, om)

        p = kc.compute_thread()

        output_manager.save(p)
