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
    install_requires=['six==1.14.0', 'h5py==2.10.0',
                      'scipy==1.4.1', 'numpy==1.18.3', 'click==7.0', 'cython==0.29.14',
                      'pandas==0.25.3', 'future==0.18.2', 'joblib==0.14.0', 'scikit-learn==0.22', 'tqdm==4.40.0',
                      'scikit-image==0.16.2', 'setuptools', 'cytoolz==0.10.1', 'ipywidgets==7.5.1',
                      'matplotlib==3.1.2', 'statsmodels==0.10.2', 'ruamel.yaml==0.16.5', 'opencv-python==4.1.2.30',
                      'pyhsmm @ git+https://github.com/mattjj/pyhsmm.git@master',
                      'pybasicbayes @ git+https://github.com/mattjj/pybasicbayes.git@master',
                      'autoregressive @ git+https://github.com/mattjj/pyhsmm-autoregressive.git@master'
                      ],
    entry_points={'console_scripts': ['moseq2-model = moseq2_model.cli:cli']},
)