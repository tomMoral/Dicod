from setuptools import setup, find_packages
setup(name='dicod',
      version='0.1.dev',
      packages=find_packages(),
      install_requires=[
          'numpy',
          'matplotlib',
          'mpi4py',
      ],
      )

