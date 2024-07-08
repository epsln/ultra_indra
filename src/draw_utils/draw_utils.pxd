import cython 
from src.data.image_dataclass cimport ImageDataclass 

cdef void line(cython.floatcomplex p0, cython.floatcomplex p1, ImageDataclass img)
