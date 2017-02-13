from __future__ import division
import numpy as np
import h5py as h5
import cPickle as pickle
import gzip
import scipy.io as sio
import copy
from train.models import ARHMM
from collections import OrderedDict
from train.util import merge_dicts, train_model
from tqdm import tqdm_notebook

# stolen from MoSeq thanks @alexbw
def enum(*sequential, **named):
    """Handy way to fake an enumerated type in Python
    http://stackoverflow.com/questions/36932/how-can-i-represent-an-enum-in-python
    """
    enums = dict(zip(sequential, range(len(sequential))), **named)
    return type('Enum', (), enums)

def parameter_scan(data_dict, parameter, values, other_parameters={}, num_iter=100, restarts=5, use_min=True):

    nparameters=len(values)
    print('User passed '+str(nparameters)+' parameter values for '+parameter)

    labels=np.empty((restarts,nparameters,len(data_dict)),dtype=object)
    #models=[[[] for i in range(nparameters)] for j in range(restarts)]
    loglikes=np.empty((restarts,nparameters),dtype=object)

    for parameter_idx, parameter_value in enumerate(tqdm_notebook(values,leave=False)):
        for itr in xrange(restarts):

            tmp_parameters=merge_dicts(other_parameters,{parameter: parameter_value})
            arhmm=ARHMM(data_dict=data_dict, **tmp_parameters)
            [arhmm,tmp_loglikes,tmp_labels]=train_model(model=arhmm,num_iter=num_iter, num_procs=1)
            loglikes[itr][parameter_idx] = tmp_loglikes

            for label_itr,tmp_label in enumerate(tmp_labels):
                labels[itr,parameter_idx,label_itr]=tmp_label

            #models[itr][parameter_idx]=copy_model(arhmm)

    return loglikes, labels

def cv_parameter_scan(data_dict, parameter, values, other_parameters={}, num_iter=100, restarts=5, use_min=False):

    nsplits=len(data_dict)
    nparameters=len(values)

    print('Will use '+str(nsplits)+' splits')
    print('User passed '+str(nparameters)+' parameter values for '+parameter)

    # by default use all the data

    if use_min:
        lens=[len(item) for item in data_dict.values()]
        use_frames=min(lens)
        print('Only using '+str(use_frames)+' per split')
        for key, item in data_dict.iteritems():
            data_dict[key]=item[:use_frames,:]

    # return the heldout likelihood, model object and labels

    heldout_ll=np.empty((restarts,nsplits,nparameters),np.float64)
    labels=np.empty((restarts,nsplits,nparameters,len(data_dict)),dtype=object)
    #models=[[[] for i in range(nparameters)] for j in range(nsplits*restarts)]

    all_keys=data_dict.keys()

    for data_idx, test_key in enumerate(tqdm_notebook(all_keys)):

        # set up the split

        train_data=OrderedDict((i,data_dict[i]) for i in all_keys if i not in test_key)
        test_data=OrderedDict([('1',data_dict[test_key])])

        for parameter_idx, parameter_value in enumerate(tqdm_notebook(values,leave=False)):
            for itr in xrange(restarts):

                tmp_parameters=merge_dicts(other_parameters,{parameter: parameter_value})
                arhmm=ARHMM(data_dict=train_data, **tmp_parameters)
                [arhmm,tmp_loglikes,tmp_labels]=train_model(model=arhmm,num_iter=num_iter, num_procs=1)
                heldout_ll[itr,data_idx,parameter_idx] = arhmm.log_likelihood(test_data['1'])

                for label_itr,tmp_label in enumerate(tmp_labels):
                    labels[itr,data_idx,parameter_idx,label_itr]=tmp_label


                #labels[itr+data_idx*(restarts)][parameter_idx]=tmp_labels
                #models[itr+data_idx*(restarts)][parameter_idx]=copy_model(arhmm)

    return heldout_ll, labels

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

def copy_model(self):
    tmp = []

    # make a deep copy of the data-less version

    for s in self.states_list:
        tmp.append(s.data)
        s.data = None

    cp=copy.deepcopy(self)

    # now put the data back in

    for s,t in zip(self.states_list, tmp):
        s.data = t

    return cp

def save_model_fit(filename, model, loglikes, labels):

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

    for i in xrange(len(labels)):
        labels_export[i]=labels[i]

    sio.savemat(filename,mdict={'labels':labels_export,'parameters':parameters,'log_likelihoods':log_likelihoods})
