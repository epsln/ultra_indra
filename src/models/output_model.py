from dataclasses import dataclass


@dataclass
class OutputModel:
    image_width: int = 1920
    image_height: int = 1080
    bounds: float = 1
    center: float = 0
    file_type: str = "jpg"
    output_directory: str = "output"
