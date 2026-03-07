from setuptools import setup, find_packages
from Cython.Build import cythonize
import os

# Define your extensions
extensions = cythonize(
    ["src/manhattan_mcp/**/*.py"],
    exclude=[
        "src/manhattan_mcp/**/__init__.py",
        "src/manhattan_mcp/**/*_test.py"
    ],
    compiler_directives={
        'language_level': "3",
        'binding': True,         # This is the critical one for FastMCP
        'embedsignature': True,  # Puts the signature in the docstring for inspect
        'always_allow_keywords': True # Ensures MCP can pass named arguments
    }
)

# This custom class tells setuptools to IGNORE the .py and .c files in the final wheel
from setuptools.command.build_py import build_py

class BuildPy(build_py):
    def find_package_modules(self, package, package_dir):
        modules = super().find_package_modules(package, package_dir)
        # Filter out .py files that have a corresponding .so being built
        return [
            (pkg, mod, file) for (pkg, mod, file) in modules 
            if not os.path.exists(file.replace(".py", ".c"))
        ]

setup(
    # ... your other setup config ...
    ext_modules=extensions,
    cmdclass={'build_py': BuildPy}, # This is the magic line
)