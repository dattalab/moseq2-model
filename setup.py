from setuptools import setup, find_packages

#testing w/o 'scikit-learn == 0.16.1','scikit-image',works okay but leaving here for reference
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
    entry_points={'console_scripts': ['MoDel = cli:cli']},
)
