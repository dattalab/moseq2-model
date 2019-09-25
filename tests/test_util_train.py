import pytest
import os
import numpy as np
import scipy
import joblib
import h5py
import copy
import joblib
from functools import partial
from collections import OrderedDict, defaultdict
from moseq2_model.train.util import train_model, get_labels_from_model,\
    whiten_all, whiten_each, zscore_each, zscore_all, get_crosslikes, slices_from_indicators, rleslices

def test_get_labels_from_model():
    model = joblib.load('tests/test_data/mock_model.p')
    cat_labels = [s.stateseq for s in model.states_list]


def test_train_model():
    model = ''
    num_iter = 100
    save_every = 1
    ncpus = 1
    cli = False