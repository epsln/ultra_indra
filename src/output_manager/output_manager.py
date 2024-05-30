import cv2
import numpy as np
import os

class OutputManager:
    def __init__(self, file_type= "jpg", output_dir = "output/"):
        self.file_type = file_type
        self.output_dir = output_dir
        self.image_count = 0
        if not os.path.isdir(self.output_dir):
            os.mkdir(self.output_dir)
        

    def save(self, image):
        self._get_filename()
        self._image_saver(image)

    def _get_filename(self):
        self.filename = os.path.join(self.output_dir, f"img_{self.image_count}.{self.file_type}")
        self.image_count += 1

    def _image_saver(self, image):
        cv2.imwrite(self.filename, image)
