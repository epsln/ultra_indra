import cv2
import logging
import numpy as np
import os
from scipy.interpolate import interp1d

_logger = logging.getLogger(__name__)

class OutputManager:
    def __init__(self, output_model):
        self.output_dir = output_model.output_directory
        self.image_height = output_model.image_height
        self.image_width = output_model.image_width
        self.bounds = output_model.bounds
        self.center = output_model.center
        self.image_count = 0
        self.file_type = output_model.file_type
        
        if not os.path.isdir(self.output_dir):
            _logger.debug(f"Creating output folder : {self.output_dir}")
            os.mkdir(self.output_dir)
    
    def process_points(self, list_points):
        img_arr = np.zeros((self.image_width, self.image_height, 3))
        xs = np.arange(self.center - self.bounds, self.center + self.bounds, self.bounds * 2/max(self.image_width, self.image_height))
        aspect_ratio = max(self.image_width, self.image_height)/min(self.image_height, self.image_width)
        #There is some way to do it without all the ifs but I can't think of anything clever right now
        #TODO: Do the clever thing
        if self.image_height > self.image_width:
            interp_func_width = interp1d(xs, np.arange(self.image_width) , fill_value = "extrapolate")
            interp_func_height = interp1d(xs * aspect_ratio, np.arange(self.image_height), fill_value = "extrapolate")
        elif self.image_height < self.image_width:
            interp_func_width = interp1d(xs * aspect_ratio, np.arange(self.image_width) , fill_value = "extrapolate")
            interp_func_height = interp1d(xs, np.arange(self.image_height), fill_value = "extrapolate")
        else:
            interp_func_width = interp1d(xs, np.arange(self.image_width) , fill_value = "extrapolate")
            interp_func_height = interp1d(xs, np.arange(self.image_height), fill_value = "extrapolate")

        for p in list_points: 
            x, y = int(interp_func_width(p.real)), int(interp_func_height(p.imag))
            if x >= 0 and x < self.image_width and y >= 0 and y < self.image_height:
                img_arr[x][y][:] = 255 
        return np.moveaxis(img_arr,0, 1)

    def save(self, image):
        postprocessed_img = self.process_points(image)
        self._get_filename()
        self._image_saver(postprocessed_img)

    def _get_filename(self):
        self.filename = os.path.join(self.output_dir, f"img_{self.image_count}.{self.file_type}")
        self.image_count += 1

    def _image_saver(self, image):
        _logger.debug(f"Saving file at : {self.filename}")
        cv2.imwrite(self.filename, image)
