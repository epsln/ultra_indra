from distutils.core import setup
from distutils.extension import Extension
import Cython.Compiler.Options
Cython.Compiler.Options.annotate = True

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
        Extension("klein_compute.tree_explorer", ["cython/tree_explorer.pyx"]),
    ]
    cmdclass.update({'build_ext': build_ext})
else:
    ext_modules += [
        Extension("klein_compute.tree_explorer", ["cython/tree_explorer.c"]),
    ]

setup(
    name='ultra_indra',
    cmdclass=cmdclass,
    ext_modules=ext_modules,
)
