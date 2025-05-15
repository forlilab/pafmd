from setuptools import setup, find_packages


setup(name="custom_openfold_utils",
      version='1.0',
      description="Utilities for running customized version of openfold optimized for generating alternative conformations",
      author="Ishan Taneja",
      author_email="itaneja@scripps.edu",
      packages=find_packages(),
      install_requires=['biopython>=1.78',
                        'pymol>=2.3.0',
                        'openmm>=7.5.0',
                        'numpy',
                        'pandas',
                        'pdbfixer',
                        'dm-tree',
                        'modelcif'], 
      license="MIT",
      classifiers=["Programming Language :: Python :: 3.8",
                   "Programming Language :: Python :: 3.9",
                   "Programming Language :: Python :: 3.10",
                   "Programming Language :: Python :: 3.11",
                   "Operating System :: Unix",
                   "Operating System :: MacOS",
                   "Topic :: Scientific/Engineering"]
)
