from src.klein_composer import KleinComposer
from src.models import FractalModel, ComputeModel, OutputModel
from src.output_manager import OutputManager
from src.recipe_manager import RecipeManager 

from fractions import Fraction
import numpy as np

import logging
import os

if __name__ == "__main__":
    rm = RecipeManager("grandma_recipe")
    fm = rm.generate(2, 2)
    logging.basicConfig(
                level=os.environ.get('LOGLEVEL', 'INFO').upper()
                )
    #fm = FractalModel(
    #    generators=np.random.uniform(-1, 1, (4, 2, 2))
    #    + 1.0j * np.random.uniform(-1, 1, (4, 2, 2)),
    #    FSA=np.array(
    #        [[1, 2, 3, 4], [1, 2, 0, 4], [1, 2, 3, 0], [0, 2, 3, 4], [1, 0, 3, 4]]
    #    ),
    #    special_fract=Fraction(0, 1),
    #)

    cm = ComputeModel(max_depth=10, num_threads=4)

    kc = KleinComposer(fm, cm)

    om = OutputModel(file_type="jpg", output_directory="output")
    output_manager = OutputManager(om)
    

    p = kc.compute_thread()

    output_manager.save(p)
