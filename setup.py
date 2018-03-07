from setuptools import setup, find_packages

# testing w/o 'scikit-learn == 0.16.1','scikit-image',works okay but leaving here for reference
# note that we need to pull in autoregressive and pybasicbayes from github,
# I've hardcorded the dependency links to use very high version numbers, hope it doesn't break anything!
# note that you will need to pass the option --process-dependency-links for this to work correctly

try:
    import numpy
except ImportError:
    import pip
    #pip.main(['install', 'numpy==1.13.0'])
    pip.main(['install', 'numpy'])

try:
    import future
except ImportError:
    import pip
    pip.main(['install', 'future'])

try:
    import six
except ImportError:
    import pip
    pip.main(['install', 'six'])

try:
    import cython
except ImportError:
    import pip
    pip.main(['install', 'cython'])

setup(
    name='moseq2_model',
    version='0.0.1',
    author='Datta Lab',
    description='Modeling for the best',
    packages=find_packages(exclude='docs'),
    platforms='any',
    python_requires='>3.4',
    install_requires=['future', 'h5py', 'click', 'numpy',
                      'pybasicbayes', 'pyhsmm',
                      'autoregressive', 'joblib',
                      'hdf5storage', 'ruamel.yaml', 'tqdm'],
    dependency_links=['git+https://github.com/mattjj/pybasicbayes.git@robust_regression#egg=pybasicbayes-1',
                      'git+https://github.com/mattjj/pyhsmm-autoregressive.git@master#egg=autoregressive-1'],
    entry_points={'console_scripts': ['moseq2-model = moseq2_model.cli:cli']},
)
