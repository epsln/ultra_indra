import cv2
import logging
import os

_logger = logging.getLogger(__name__)


class OutputManager:
    def __init__(self, output_model):
        self.output_dir = output_model.output_directory
        self.image_count = 0
        self.file_type = output_model.file_type

        if not os.path.isdir(self.output_dir):
            os.mkdir(self.output_dir)

    def save(self, image):
        self._get_filename()
        self._image_saver(image)

    def _get_filename(self):
        self.filename = os.path.join(
            self.output_dir, f"img_{self.image_count}.{self.file_type}"
        )
        self.image_count += 1
        _logger.info(f"Saving {self.filename}")

    def _image_saver(self, image):
        cv2.imwrite(self.filename, image.T)
