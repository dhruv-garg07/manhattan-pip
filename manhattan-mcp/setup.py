from setuptools import setup, Extension
from Cython.Build import cythonize
import os
from pathlib import Path

# 1. OS-Agnostic File Discovery
src_dir = Path("src/manhattan_mcp")
py_files = []

# Walk through the directory and grab all .py files
for file_path in src_dir.rglob("*.py"):
    # Exclude __init__.py and tests
    if file_path.name != "__init__.py" and not file_path.name.endswith("_test.py"):
        py_files.append(str(file_path))

# 2. Cythonize the discovered files
extensions = cythonize(
    py_files,
    compiler_directives={
        'language_level': "3",
        'binding': True,
        'embedsignature': True,
        'always_allow_keywords': True
    }
)

# 3. The "Kill Switch" to prevent .py files in the final wheel
from setuptools.command.build_py import build_py

class BuildPy(build_py):
    def find_package_modules(self, package, package_dir):
        modules = super().find_package_modules(package, package_dir)
        # Filter out .py files so only the compiled .so / .pyd binaries make it into the wheel
        return [
            (pkg, mod, file) for (pkg, mod, file) in modules 
            # We check the file extension instead of looking for the .c file,
            # which avoids race conditions during the build sequence.
            if not file.endswith(".py") or file.endswith("__init__.py")
        ]

setup(
    ext_modules=extensions,
    cmdclass={'build_py': BuildPy},
)