from __future__ import division
import numpy as np
import h5py as h5
import cPickle as pickle
import gzip
import scipy.io as sio
import copy
from collections import OrderedDict
from train.util import merge_dicts

# sort data into n splits, farm each split w/ bsub in one version

def cv_parameter_scan(data_dict, config_file, output_dir, use_min=True):

    # number of keys in dictionary defines splits (could think of more complicated schemes)

    print('Will use '+len(data_dict)+' splits')

    if use_min:
        lens=[len(item) for item in data_dict.values()]
        use_frames=min(lens)
        print('Only using '+use_frames+'per split')
        for key, item in data_dict.iteritems():
            data_dict[key]=item[:use_frames,:]

    # config file yaml?

    # consider this a gateway from a command line interface, data_dict points to a pickle
    # and yaml specifies parameter and scan values (could do fminsearch-like thing here)


    pass

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
            tmp.append(s.data)
            s.data = None
        cp=copy.deepcopy(self)
        for s,t in zip(self.states_list, tmp):
            s.data = t
        return cp

    with gzip.open(filename, 'w') as outfile:
        pickle.dump({'model': copy_model(model), 'loglikes': loglikes, 'labels': labels},
                    outfile, protocol=-1)

def export_model_to_matlab(filename, model, log_likelihoods, labels):

    trans_dist=model.trans_distn
    init_obs_dist=model.init_emission_distn.hypparams

    parameters= {
        'ar_mat':[obs.A for obs in model.obs_distns],
        'sig':[obs.sigma for obs in model.obs_distns],
        'kappa':trans_dist.kappa,
        'gamma':trans_dist.gamma,
        'alpha':trans_dist.alpha,
        'num_states':trans_dist.N,
        'nu_0':init_obs_dist['nu_0'],
        'sigma_0':init_obs_dist['sigma_0'],
        'kappa_0':init_obs_dist['kappa_0'],
        'nlags':model.nlags,
        'mu_0':init_obs_dist['mu_0']
    }

    # use savemat to save in a format convenient for dissecting in matlab

    # prepend labels with -1 to account for lags, also put into Dict to convert to a cell array

    labels=[np.hstack((np.full((label.shape[0],model.nlags),-1),label)) for label in labels]
    labels_export=np.empty(len(labels),dtype=object)

    for i in xrange(0,len(labels)):
        labels_export[i]=labels[i]

    sio.savemat(filename,mdict={'labels':labels_export,'parameters':parameters,'log_likelihoods':log_likelihoods})

    pass
