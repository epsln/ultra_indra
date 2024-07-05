from distutils.core import setup
from distutils.extension import Extension
import Cython.Compiler.Options

from Cython.Build import cythonize
try:
    from Cython.Distutils import build_ext
except ImportError:
    use_cython = False
else:
    use_cython = True

cmdclass = {}
ext_modules = []

if use_cython:
    ext_modules += [
        Extension("src.klein_dataclass.klein_dataclass", ["src/klein_dataclass/klein_dataclass.pyx"]),
        Extension("src.tree_explorer.tree_explorer", ["src/tree_explorer/tree_explorer.pyx"]),
        Extension("src.draw_utils.draw_utils", ["src/draw_utils/draw_utils.pyx"]),
    ]
    cmdclass.update({'build_ext': build_ext})
else:
    ext_modules += [
        Extension("klein_compute.tree_explorer", ["src/tree_explorer/tree_explorer.c"]),
        Extension("klein_compute.draw_utils", ["src/draw_utils/draw_utils.c"]),
    ]

setup(
    name='ultra_indra',
    cmdclass=cmdclass,
    ext_modules=ext_modules,
)
