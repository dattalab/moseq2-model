from __future__ import division
import numpy as np
from functools import partial
from collections import OrderedDict
from tqdm import tqdm, tqdm_notebook
from kinect_modeling.util import progressbar

# TODO: simple function for cross-validation optimization of parameters
#def cv_parameter_scan()


# based on moseq by @mattjj and @alexbw
def train_model(model, num_iter=100, save_every=1, num_procs=1, cli=False, **kwargs):

    # per conversations w/ @mattjj, the fast class of models use openmp no need
    # for "extra" parallelism

    log_likelihoods = []
    labels = []

    for itr in progressbar(range(num_iter),cli=cli,**kwargs):
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
def whiten_each(data_dict, center=True):
    for k,v in data_dict.items():
        tmp_dict=whiten_all(OrderedDict([(k,v)]),center=center)
        data_dict[k]=tmp_dict[k]

    return data_dict
    #return OrderedDict((k, whiten_all(OrderedDict([k,v]), center=center)) for k, v in data_dict.items())

def parameter_scan(data_dict, parameter, values, other_parameters=dict(),
                   num_iter=100, restarts=5, use_min=True):

    nparameters=len(values)
    print('User passed '+str(nparameters)+' parameter values for '+parameter)

    labels=np.empty((restarts,nparameters,len(data_dict)),dtype=object)
    #models=[[[] for i in range(nparameters)] for j in range(restarts)]
    loglikes=np.empty((restarts,nparameters),dtype=object)

    for parameter_idx, parameter_value in enumerate(progressbar(values,leave=False)):
        for itr in xrange(restarts):

            tmp_parameters=merge_dicts(other_parameters,{parameter: parameter_value})
            arhmm=ARHMM(data_dict=data_dict, **tmp_parameters)
            [arhmm,tmp_loglikes,tmp_labels]=train_model(model=arhmm,num_iter=num_iter, num_procs=1)
            loglikes[itr][parameter_idx] = tmp_loglikes

            for label_itr,tmp_label in enumerate(tmp_labels):
                labels[itr,parameter_idx,label_itr]=tmp_label

            #models[itr][parameter_idx]=copy_model(arhmm)

    return loglikes, labels


def cv_parameter_scan(data_dict, parameter, values, other_parameters=dict(),
                      num_iter=100, restarts=5, use_min=False):

    nsplits = len(data_dict)
    nparameters = len(values)

    print 'Will use '+str(nsplits)+' splits'
    print 'User passed '+str(nparameters)+' parameter values for '+parameter

    # by default use all the data

    if use_min:
        lens=[len(item) for item in data_dict.values()]
        use_frames=min(lens)
        print('Only using '+str(use_frames)+' per split')
        for key, item in data_dict.iteritems():
            data_dict[key]=item[:use_frames,:]

    # return the heldout likelihood, model object and labels

    heldout_ll=np.empty((restarts,nsplits,nparameters), np.float64)
    labels=np.empty((restarts,nsplits,nparameters,len(data_dict)),dtype=object)
    #models=[[[] for i in range(nparameters)] for j in range(nsplits*restarts)]

    all_keys=data_dict.keys()

    for data_idx, test_key in enumerate(progressbar(all_keys)):

        # set up the split

        train_data=OrderedDict((i,data_dict[i]) for i in all_keys if i not in test_key)
        test_data=OrderedDict([('1',data_dict[test_key])])

        for parameter_idx, parameter_value in enumerate(progressbar(values,leave=False)):
            for itr in xrange(restarts):

                tmp_parameters=merge_dicts(other_parameters,{parameter: parameter_value})
                arhmm=ARHMM(data_dict=train_data, **tmp_parameters)
                [arhmm, _, tmp_labels]=train_model(model=arhmm, num_iter=num_iter, num_procs=1)
                heldout_ll[itr,data_idx,parameter_idx] = arhmm.log_likelihood(test_data['1'])

                for label_itr,tmp_label in enumerate(tmp_labels):
                    labels[itr, data_idx, parameter_idx, label_itr] = tmp_label


                #labels[itr+data_idx*(restarts)][parameter_idx]=tmp_labels
                #models[itr+data_idx*(restarts)][parameter_idx]=copy_model(arhmm)

    return heldout_ll, labels
