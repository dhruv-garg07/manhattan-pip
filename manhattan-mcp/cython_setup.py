from setuptools import setup, find_packages
from Cython.Build import cythonize

# We target the actual package folder inside src
extensions = cythonize(
    ["src/manhattan_mcp/**/*.py"],
    exclude=[
        "src/manhattan_mcp/**/__init__.py",
        "src/manhattan_mcp/**/*_test.py"
    ],
    compiler_directives={'language_level': "3"}
)

setup(
    name="manhattan_mcp",
    version="0.1.0",
    # This tells setuptools that the 'manhattan_mcp' package is inside the 'src' directory
    package_dir={"": "src"},
    packages=find_packages(where="src"),
    ext_modules=extensions,
    zip_safe=False,
)