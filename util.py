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

def cv_parameter_scan(data_dict, config_file, output_file, parameter, values, restarts=5, use_min=True):

    nsplits=len(data_dict)
    nparameters=len(values)

    print('Will use '+str(nsplits)+' splits')
    print('User passed '+str(nparameters)+' parameter values')

    if use_min:
        lens=[len(item) for item in data_dict.values()]
        use_frames=min(lens)
        print('Only using '+use_frames+'per split')
        for key, item in data_dict.iteritems():
            data_dict[key]=item[:use_frames,:]

        # config file yaml?

    heldout_ll=np.empty(nsplits*restarts,nparameters)
    all_keys=data_dict.keys()

    for data_idx, test_key in enumerate(all_keys):

        # set up the split

        train_keys=[x for x in all_keys if x not in test_key]
        train_data=OrderedDict((i,features_trim[i]) for i in train_keys)
        test_data=OrderedDict()
        test_data['1']=features_trim[test_key]

        for parameter_idx, parameter_values in enumerate(values):

            for itr in xrange(0,restarts):

                arhmm=ARHMM(data_dict=train_data, parameter=parameter_value)
                [tmp_arhmm,tmp_loglikes,tmp_labels]=train_model(model=model,num_iter=100, num_procs=1)
                tmp_heldout_ll=tmp_arhmm.log_likelihood(test_data['1'])
                heldout_ll[itr+(data_idx*(nrestarts-1))][parameter_idx] = tmp_heldout_ll


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
