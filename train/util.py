from __future__ import division
import numpy as np
from pyhsmm.util.text import progprint_xrange
from functools import partial
from collections import OrderedDict

# taken from moseq by @mattjj and @alexbw
def train_model(model, num_iter=100, save_every=1, num_procs=8):
    def resample(model):
        #model.resample_model(num_procs=num_procs)
        model.resample_model()
        return model

    def snapshot(model):
        return model.log_likelihood(), get_labels_from_model(model)

    print(str(num_iter))
    model_samples = [resample(model) for itr in progprint_xrange(num_iter)]
    saved_samples = [snapshot(model) for itr, model in enumerate(model_samples) if eq_mod(itr, -1, save_every)]

    loglikes, labels = zip(*saved_samples)

    # different format for storing labels

    for i in xrange(0,itr):
        labels_cat[i]=np.array([tmp[i] for tmp in labels])

    return model, loglikes, labels_cat
    #return model

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


def eq_mod(a, b, c):
    return (a % c) == (b % c)
