from setuptools import setup, find_packages

# testing w/o 'scikit-learn == 0.16.1','scikit-image',works okay but leaving here for reference
# note that we need to pull in autoregressive and pybasicbayes from github,
# I've hardcorded the dependency links to use very high version numbers, hope it doesn't break anything!
# note that you will need to pass the option --process-dependency-links for this to work correctly

setup(
    name='moseq2_model',
    version='0.4.0',
    author='Datta Lab',
    description='Modeling for the best',
    packages=find_packages(exclude='docs'),
    platforms='any',
    python_requires='>=3.6',
    install_requires=[
        'pyhsmm @ git+https://github.com/mattjj/pyhsmm.git@master',
        'pybasicbayes @ git+https://github.com/mattjj/pybasicbayes.git@master',
        'autoregressive @ git+https://github.com/mattjj/pyhsmm-autoregressive.git@master'
    ],
    entry_points={'console_scripts': ['moseq2-model = moseq2_model.cli:cli']},
)