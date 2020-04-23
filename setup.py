import os
import sys
import subprocess
from setuptools import setup, find_packages

os.system('export CC="$(which gcc-7)"')
os.system('export CXX="$(which g++-7)"')

def install(package):
    subprocess.call([sys.executable, "-m", "pip", "install", package])


try:
    import numpy
except ImportError:
    install('numpy')

try:
    import future
except ImportError:
    install('future')

try:
    import six
except ImportError:
    install('six')

try:
    import cython
except ImportError:
    install('cython')


setup(
    name='moseq2_model',
    version='0.4.0',
    author='Datta Lab',
    description='Modeling for the best',
    packages=find_packages(exclude='docs'),
    include_package_data=True,
    platforms='any',
    python_requires='>=3.6',
    setup_requires=['numpy', "future", "six"],
    install_requires=['six', 'h5py', 'scipy', 'numpy', 'click', 'cython',
                      'pandas', 'future', 'joblib', 'scikit-learn',
                      'scikit-image', 'setuptools', 'cytoolz', 'ipywidgets',
                      'matplotlib', 'statsmodels', 'ruamel.yaml', 'opencv-python',
                      'pyhsmm @ git+https://github.com/mattjj/pyhsmm.git@master',
                      'pybasicbayes @ git+https://github.com/mattjj/pybasicbayes.git@master',
                      'autoregressive @ git+https://github.com/mattjj/pyhsmm-autoregressive.git@master'
                      ],
    entry_points={'console_scripts': ['moseq2-model = moseq2_model.cli:cli']},
)