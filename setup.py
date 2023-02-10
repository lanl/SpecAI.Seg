from setuptools import setup, find_packages
from Cython.Build import cythonize

setup(name="HSIsuqs",    version="0.1.0",    packages=find_packages(),
ext_modules=cythonize(["HSIsuqs/models/_slic.pyx"],
                       compiler_directives={'language_level' : "3"}))