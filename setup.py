import os
import sys
import codecs
import subprocess
from setuptools import setup, find_packages


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


def read(rel_path):
    here = os.path.abspath(os.path.dirname(__file__))
    with codecs.open(os.path.join(here, rel_path), 'r') as fp:
        return fp.read()

def get_version(rel_path):
    for line in read(rel_path).splitlines():
        if line.startswith('__version__'):
            delim = '"' if '"' in line else "'"
            return line.split(delim)[1]
    else:
        raise RuntimeError("Unable to find version string.")


setup(
    name='moseq2_model',
    version=get_version("moseq2_model/__init__.py"),
    author='Datta Lab',
    description='Modeling for the best',
    packages=find_packages(exclude='docs'),
    include_package_data=True,
    platforms='any',
    python_requires='>=3.6',
    install_requires=['six==1.15.0', 'h5py==2.10.0',
                      'scipy==1.3.2', 'numpy==1.18.3', 'click==7.0', 'cython==0.29.14',
                      'pandas==1.0.5', 'future==0.18.2', 'joblib==0.15.1',
                      'scikit-image==0.16.2', 'setuptools', 'cytoolz==0.10.1', 'tqdm==4.40.0',
                      'matplotlib==3.1.2', 'statsmodels==0.10.2', 'ruamel.yaml==0.16.5', 'opencv-python==4.1.2.30',
                      'pyhsmm @ git+https://github.com/mattjj/pyhsmm.git@master',
                      'pybasicbayes @ git+https://github.com/mattjj/pybasicbayes.git@master',
                      'autoregressive @ git+https://github.com/dattalab/pyhsmm-autoregressive.git@master'
                      ],
    entry_points={'console_scripts': ['moseq2-model = moseq2_model.cli:cli']},
)
