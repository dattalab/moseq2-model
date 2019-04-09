import subprocess, sys
from setuptools import setup, find_packages

# testing w/o 'scikit-learn == 0.16.1','scikit-image',works okay but leaving here for reference
# note that we need to pull in autoregressive and pybasicbayes from github,
# I've hardcorded the dependency links to use very high version numbers, hope it doesn't break anything!
# note that you will need to pass the option --process-dependency-links for this to work correctly

install = lambda pkg: subprocess.check_call([sys.executable, '-m', 'pip', 'install', pkg])

try:
    import numpy
except ImportError:
    #pip.main(['install', 'numpy==1.13.0'])
    #pip.main(['install', 'numpy'])
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
    version='0.1.3',
    author='Datta Lab',
    description='Modeling for the best',
    packages=find_packages(exclude='docs'),
    platforms='any',
    python_requires='>=3.6',
    install_requires=['future', 'h5py', 'click', 'numpy==1.14.5',
                      'pyhsmm', 'joblib>=0.13.1',
                      'hdf5storage', 'ruamel.yaml>=0.15.0', 'tqdm',
                      'pybasicbayes @ git+https://github.com/mattjj/pybasicbayes.git@master',
                      'autoregressive @ git+https://github.com/mattjj/pyhsmm-autoregressive.git@master'],
    entry_points={'console_scripts': ['moseq2-model = moseq2_model.cli:cli']},
)
