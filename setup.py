from setuptools import setup, find_packages

from pathlib import Path
this_directory = Path(__file__).parent
long_description = (this_directory / "README.md").read_text()

setup(
    name="specaiseg",
    version="1.0.7",    
    packages=find_packages(),
    install_requires = [
        'numpy',
        'pandas',
        'torch',
        'scikit-image',
        'scipy',
        'scikit-learn',
        'tqdm',
        'spectral',
        'matplotlib'
    ],
    package_data={'specaiseg' : ['specaiseg/data_raw/datasets.csv']},
    description='Code for doing hyperspectral image segmentation and uncertainty quantification.',
    long_description=long_description,
    long_description_content_type='text/markdown',
    url='https://github.com/lanl/specaiseg',
    author='Scout Cabe Jarman',
    author_email='scoutjarman@yahoo.com',
    license='GPLv3',
    classifiers=[
        'Development Status :: 3 - Alpha',
        'Intended Audience :: Science/Research',
        'Intended Audience :: Education',
        'Topic :: Scientific/Engineering :: Image Processing',
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
        'Topic :: Scientific/Engineering :: Visualization',
        'License :: OSI Approved :: GNU General Public License v3 (GPLv3)',
        'Programming Language :: Python :: 3' ,
        'Environment :: GPU :: NVIDIA CUDA',
        'Natural Language :: English',
        'Programming Language :: Cython',
        'Programming Language :: Python :: 3'],
    keywords='Hyperspectral Image Analysis Image Segmentation',
    python_requires='>=3',
    )