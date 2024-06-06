from dataclasses import dataclass
import numpy as np

class OutputModel:
    width: int = 1920
    height: int = 1080 
    file_type: str = "jpg" 
    output_directory: str = "output" 
