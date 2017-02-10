from __future__ import division
import numpy as np
from pyhsmm.util.text import progprint_xrange
from functools import partial
from collections import OrderedDict

# TODO: simple function for cross-validation optimization of parameters
#def cv_parameter_scan()

# taken from moseq by @mattjj and @alexbw
def train_model(model, num_iter=100, save_every=1, num_procs=8):

    log_likelihoods=[]
    labels=[]

    for itr in progprint_xrange(num_iter):
        model.resample_model()
        log_likelihoods.append(model.log_likelihood())
        labels.append([s.copy() for s in model.stateseqs])

    # different format for storing labels
    labels_cat=[]

    for i in xrange(0,len(labels[0])):
        labels_cat.append(np.array([tmp[i] for tmp in labels]))

    return model, log_likelihoods, labels_cat

# simple function for grabbing model labels across the dict

def get_labels_from_model(model):
    cat_labels=[s.stateseq for s in model.states_list]

    return cat_labels

# taken from moseq by @mattjj and @alexbw
def whiten_all(data_dict, center=True):
    non_nan = lambda x: x[~np.isnan(np.reshape(x, (x.shape[0], -1))).any(1)]
    meancov = lambda x: (x.mean(0), np.cov(x, rowvar=False, bias=1))
    contig = partial(np.require, dtype=np.float64, requirements='C')

    mu, Sigma = meancov(np.concatenate(map(non_nan, data_dict.values())))
    L = np.linalg.cholesky(Sigma)

    offset = 0. if center else mu
    apply_whitening = lambda x:  np.linalg.solve(L, (x-mu).T).T + offset

    return OrderedDict((k, contig(apply_whitening(v))) for k, v in data_dict.items())

# taken from moseq by @mattjj and @alexbw
def merge_dicts(base_dict, clobbering_dict):
    return dict(base_dict, **clobbering_dict)
