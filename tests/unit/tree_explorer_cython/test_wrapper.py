import importlib
import sys

# list of Cython modules containing tests
cython_test_modules = ["test_tree_explorer"]

for mod in cython_test_modules:
    try:
        # For each callable in `mod` with name `test_*`,
        # set the result as an attribute of this module.
        mod = importlib.import_module(mod)
        for name in dir(mod):
            item = getattr(mod, name)
            if callable(item) and name.startswith("test_"):
                setattr(sys.modules[__name__], name, item)
    except ImportError:
        pass
