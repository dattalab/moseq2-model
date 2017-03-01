from __future__ import division
import numpy as np
from functools import partial
from collections import OrderedDict
from tqdm import tqdm, tqdm_notebook
import warnings

# TODO: simple function for cross-validation optimization of parameters
#def cv_parameter_scan()

# based on moseq by @mattjj and @alexbw
def train_model(model, num_iter=100, save_every=1, num_procs=1, cli=False, disable=False, position=0):

    # per conversations w/ @mattjj, the fast class of models use openmp no need
    # for "extra" parallelism

    warnings.filterwarnings("ignore", category=np.VisibleDeprecationWarning)


    try:

        log_likelihoods = []
        labels = []

        for itr in progressbar(range(num_iter),leave=False,cli=cli,position=position, disable=disable):
            model.resample_model(num_procs)
            if np.mod(itr+1,save_every)==0:
                log_likelihoods.append(model.log_likelihood())
                seq_list=[s.stateseq for s in model.states_list]
                for seq_itr in xrange(len(seq_list)):
                    seq_list[seq_itr]=np.append(np.repeat(-5,model.nlags),seq_list[seq_itr])
                labels.append(seq_list)

        labels_cat=[]

        for i in xrange(len(labels[0])):
            labels_cat.append(np.array([tmp[i] for tmp in labels],dtype=np.int16))

    except Exception as e:

        print(str(e))
        log_likelihoods = np.nan
        labels_cat = np.nan

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

#
def progressbar(*args, **kwargs):

    cli=kwargs.pop('cli',False)

    if cli==True:
        return tqdm(*args, **kwargs)
    else:
        try:
            return tqdm_notebook(*args, **kwargs)
        except:
            return tqdm(*args, **kwargs)
