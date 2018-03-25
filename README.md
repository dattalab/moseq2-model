# MoSeq2-model [![Build Status](https://travis-ci.com/dattalab/moseq2_model.svg?token=gvoikVySDHEmvHT7Dbed&branch=master)](https://travis-ci.com/dattalab/moseq2_model) [![codecov](https://codecov.io/gh/dattalab/moseq2_model/branch/master/graph/badge.svg?token=q9xxVhps5o)](https://codecov.io/gh/dattalab/moseq2_model)

Welcome to moseq2-model, a package of convenience functions for submitting data extracted using moseq2 for modeling.  This package can be run entirely on a local machine, can be run locally to orchestra a Kubernetes job, or can be run on a Slurm HPC cluster.  You will want to use either the Kubernetes or Slurm functionality to do large parameter scans.  Each one of these situations is described below.

## Table of Contents  

- [Installation (local)](#python-packages)
- [Installation (Kubernetes)](README_kubernetes.md)
- [Installation (Slurm)](README_slurm.md)
- [Usage (model training, local)](#model-training)
- [Usage (parameter scanning, Kubernetes)](README_kubernetes.md#parameter-scan-kubernetes)
- [Usage (parameter scanning, Slurm)](README_slurm.md#parameter-scan-slurm)
- [Contributing](#contributing)

## Installation of python packages

### Python packages

Moseq2-model has been tested with Python 3.6.  All requirements can be installed using pip, though it's recommended you create a new conda environment to ensure the dependencies do not conflict with any software you may have currently installed.

```sh
conda create -n moseq2_model python=3.6
source activate moseq2_model
git clone https://github.com/dattalab/moseq2_model.git
cd moseq2_model/
pip install -e . --process-dependency-links
```

If you just want to use these tools locally, you're done.  


## Usage

### Data format (pickle)

If you save your PCs with the extension `.p, .p.z, .pkl`, then it is assumed that your data is stored as a pickle saved through `joblib.dump`.  You may use one of the two following formats for your PCs,

1. An ordered dictionary (from `collections.OrderedDict`) with key value pairs, where each value is a 2D numpy array that is `nframes x pcs`, and each key specifies a different session (or experiment, doesn't matter).  Here, the keys are completely arbitrary, but your results are returned in the same order, which is why we use an ordered dictionary.

```python
from collections import OrderedDict
import numpy
import joblib
fake_data=OrderedDict()
fake_data['session1']=numpy.random.randn(1000,10)
fake_data['session2']=numpy.random.randn(1000,10)
joblib.dump(fake_data, 'myfakedata.p', compress=True)
```

2. An ordered dictionary where each value is a `tuple', where the first value is an ndarray with the PCs and the second value is a string specifying the group.  This will be used to specify separate transition matrices when training the model.

```python
from collections import OrderedDict
import numpy
import joblib
fake_data=OrderedDict()
fake_data[1]=(np.random.randn(1000,10),'saline')
fake_data[2]=(np.random.randn(1000,10),'drug')
fake_data[3]=(np.random.randn(1000,10),'saline')
joblib.dump(fake_data, 'myfakedata.p', compress=True)
```

Now keys `1` and `3` will use the same transition matrix, and `2` will use a separate one.

### Data format (MATLAB)

If you are exporting your PCs from a MATLAB environment, use the `.mat` extension, and save your PCs and groups as separate cell arrays.

```matlab
features={}
features{1}=randn(1000,10)
features{2}=randn(1000,10)
groups{1}='saline'
groups{2}='drug'
save('myfakedata.mat','fake_data','groups','-v7.3')
```

By default `moseq2-model` assumes that your pcs live in `features` and your groups in `groups`.  Use the `-v7.3` flag to make sure you use an hdf5-compatible format.

### Model training (local)

To train a model use the `learn-model` command so, e.g.

```sh
moseq2-model learn-model myfakedata.p myresults.p.z
```

This will train a model using all default parameters, and will store the results in `myresults.p.z`.  To explore the myriad options,

```sh
moseq2-model learn-model --help
```

### Parameter scan (local)

The simplest way to do a parameter scan is in a bash loop...


## Support

## Contributing
