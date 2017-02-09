from __future__ import division
import numpy as np
import h5py as h5
import cPickle as pickle
import gzip
from collections import OrderedDict

# grab matlab data

def load_data_from_matlab(filename,varname="features",pcs=10):

    f=h5.File(filename)
    score_tmp=f[varname]
    data_dict=OrderedDict()

    for i in xrange(0,len(score_tmp)):
        tmp=f[score_tmp[i][0]]
        score_to_add=tmp.value
        data_dict[str(i+1)]=score_to_add[:pcs,:].T

    return data_dict

# per Scott's suggestion

def save_model_fit(filename, model, loglikes, labels):

    def copy_model(self):
        tmp = []
        for s in self.states_list:
            tmp.append(self.data)
            s.data = None
        cp=copy.deepcopy(self)
        for s,t in zip(self.states_list, tmp):
            s.data = t
        return cp

    with gzip.open(filename, 'w') as outfile:
        pickle.dump({'model': copy_model(model), 'loglikes': loglikes, 'labels': labels},
                    outfile, protocol=-1)

def export_model_to_matlab(filename, model, loglikes, labels)

    # save initialization parameters along with AR matrices, noise matrix
    
    # export all the stuff we need for analyzing data downstream in MATLAB
