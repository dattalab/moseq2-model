from setuptools import setup, find_packages

#testing w/o 'scikit-learn == 0.16.1','scikit-image',works okay but leaving here for reference
# note that we need to pull in autoregressive and pybasicbayes from github,
# I've hardcorded the dependency links to use very high version numbers, hope it doesn't break anything!
# note that you will need to pass the option --process-dependency-links for this to work correctly
setup(
    name='kinect_modeling',
    version='0.0.1',
    author='Datta Lab',
    description='Modeling for the best',
    license='Crapl',
    packages=find_packages(exclude='docs'),
    platforms='any',
    install_requires=['h5py',
        'click', 'pybasicbayes', 'pyhsmm', 'autoregressive', 'joblib', 'numpy == 1.11.3',
         'hdf5storage', 'ruamel.yaml','tqdm'],
    dependency_links=['git+https://github.com/mattjj/pybasicbayes.git@robust_regression#egg=pybasicbayes-1',
                      'git+https://github.com/mattjj/pyhsmm-autoregressive.git@master#egg=autoregressive-1'],
    entry_points={'console_scripts': ['MoDel = cli:cli']},
)
