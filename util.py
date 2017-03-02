from __future__ import division
import numpy as np
import h5py as h5
import joblib
import scipy.io as sio
import copy
import ruamel.yaml as yaml
import itertools
from train.models import ARHMM
from collections import OrderedDict
from train.util import merge_dicts, train_model, progressbar

# stolen from MoSeq thanks @alexbw
def enum(*sequential, **named):
    """Handy way to fake an enumerated type in Python
    http://stackoverflow.com/questions/36932/how-can-i-represent-an-enum-in-python
    """
    enums = dict(zip(sequential, range(len(sequential))), **named)
    return type('Enum', (), enums)


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

# grab matlab data

def load_pcs(filename,var_name,npcs=10):

    # TODO: trim pickles down to right number of pcs

    if filename.endswith('.mat'):
        data_dict=load_data_from_matlab(filename,var_name,npcs)
    elif filename.endswith('.z') or filename.endswith('.pkl') or filename.endswith('.p'):
        data_dict=joblib.load(filename)
    elif filename.endswith('.h5'):
        from moseq.util import load_field_from_hdf
        data_dict = load_field_from_hdf(filename, 'data')
    else:
        raise ValueError('Did understand filetype')

    return data_dict


def save_dict(filename,obj_to_save=None,print_message=False):

    # we gotta switch to lists here my friend, create a file with multiple
    # pickles, only load as we need them

    if filename.endswith('.mat'):
        if not print_message:
            print('Saving MAT file '+filename)
            sio.savemat(filename,mdict=obj_to_save)
        else:
            print('Will save MAT file '+filename)
    elif filename.endswith('.z'):
        # pickle it
        if not print_message:
            print('Saving compressed pickle '+filename)
            joblib.dump(obj_to_save, filename, compress=3)
        else:
            printctivity('Will save compressed pickle '+filename)
    elif filename.endswith('.pkl') | filename.endswith('.p'):
        # pickle it
        if not print_message:
            print('Saving pickle '+filename)
            joblib.dump(obj_to_save, filename, compress=0)
        else:
            print('Will save piclke '+filename)
    else:
        raise ValueError('Did understand filetype')


def load_data_from_matlab(filename,var_name="features",npcs=10):

    f=h5.File(filename)
    score_tmp=f[var_name]
    data_dict=OrderedDict()

    for i in xrange(0,len(score_tmp)):
        tmp=f[score_tmp[i][0]]
        score_to_add=tmp.value
        data_dict[str(i+1)]=score_to_add[:npcs,:].T

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
    joblib.dump({'model': copy_model(model), 'loglikes': loglikes, 'labels': labels})

def get_parameters_from_model(model,save_ar=True):

    trans_dist=model.trans_distn
    init_obs_dist=model.init_emission_distn.hypparams

    parameters= {
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

    if save_ar:
        parameters['ar_mat']=[obs.A for obs in model.obs_distns]
        parameters['sig']=[obs.sigma for obs in model.obs_distns]

    return parameters


# read in user yml file for mpi jobs
def read_cli_config(filename):

    with open(filename, 'r') as f:
        config = yaml.load(f.read(), Loader=yaml.Loader)

    cfg={
        'worker_dicts':None,
        'scan_parameter':None,
        'scan_value':None,
        'other_parameters':{}
    }

    if 'scan_settings' in config:

        cfg=merge_dicts(cfg,config['scan_settings'])
        cfg['scan_values'] = []
        cfg['worker_dicts']= []

        if type(cfg['scan_parameter']) is list:
            for use_parameter,use_range,use_scale in zip(cfg['scan_parameter'],cfg['scan_range'],cfg['scan_scale']):
                if use_scale=='log':
                    cfg['scan_values'].append(np.logspace(*use_range))
                elif use_scale=='linear':
                    cfg['scan_values'].append(np.linspace(*use_range))

            for itr_values in itertools.product(*cfg['scan_values']):
                new_dict = {}
                for param,value in zip(cfg['scan_parameter'],itr_values):
                    new_dict[param]=value
                cfg['worker_dicts'].append(new_dict)
        else:
            if cfg['scan_scale']=='log':
                cfg['scan_values'].append(np.logspace(*cfg['scan_range']))
            elif cfg['scan_scale']=='linear':
                cfg['scan_values'].append(np.linspace(*cfg['scan_range']))

            for value in cfg['scan_values'][0]:
                new_dict = {
                    cfg['scan_parameter']: value
                }
                cfg['worker_dicts'].append(new_dict)

    other_parameters={}
    print(cfg)
    if 'parameters' in config.keys():
        cfg['other_parameters']=config['parameters']

    if type(cfg['scan_parameter']) is list:
        for param,values in zip(cfg['scan_parameter'],cfg['scan_values']):
            print('Will scan parameter '+param)
            print('Will scan value '+str(values))
    else:
        print('Will scan parameter '+cfg['scan_parameter'])
        print('Will scan value '+str(cfg['scan_values'][0]))

    return cfg

# credit to http://stackoverflow.com/questions/14000893/specifying-styles-for-portions-of-a-pyyaml-dump
class blockseq( dict ): pass
def blockseq_rep(dumper, data):
    return dumper.represent_mapping( u'tag:yaml.org,2002:map', data, flow_style=False )

class flowmap( dict ): pass
def flowmap_rep(dumper, data):
    return dumper.represent_mapping( u'tag:yaml.org,2002:map', data, flow_style=True )
