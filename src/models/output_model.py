from dataclasses import dataclass


@dataclass
class OutputModel:
    image_dim: tuple = (1920, 1080)
    z_min: complex = -1 - 1j
    z_max: complex = +1 + 1j
    file_type: str = "jpg"
    output_directory: str = "output"
