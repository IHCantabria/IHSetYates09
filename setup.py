from setuptools import setup, find_packages, Extension
from Cython.Build import cythonize
import numpy

# Definindo as extensões Cython
extensions = [
    Extension("IHSetYates09.fast_simulator", ["IHSetYates09/fast_simulator.pyx"], include_dirs=[numpy.get_include()]),
]

setup(
    name='IHSetYates09',
    version='1.1.7',
    packages=find_packages(),
    include_package_data=True,
    ext_modules=cythonize(extensions),  # Adiciona a compilação Cython
    install_requires=[
        'numpy',
        'xarray',
        'numba',
        'datetime',
        'spotpy',
        'IHSetCalibration @ git+https://github.com/IHCantabria/IHSetCalibration.git',
        'fast_optimization @ git+https://github.com/defreitasL/fast_optimization.git'
    ],
    author='Lucas de Freitas Pereira',
    author_email='lucas.defreitas@unican.es',
    description='IH-SET Yates et al. (2009)',
    url='https://github.com/IHCantabria/IHSetYates09',
    classifiers=[
        'Development Status :: 3 - Alpha',
        'Intended Audience :: Developers',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
    ],
)
