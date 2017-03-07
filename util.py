from __future__ import division
import numpy as np
import h5py as h5
import joblib
import scipy.io as sio
import copy
import ruamel.yaml as yaml
import itertools
import os
import subprocess
import re
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
def read_cli_config(filename,suppress_output=False):

    with open(filename, 'r') as f:
        config = yaml.load(f.read(), Loader=yaml.Loader)

    cfg={
        'worker_dicts':None,
        'scan_parameter':None,
        'scan_values':None,
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

    cfg['other_parameters']={}

    if 'parameters' in config.keys():
        cfg['other_parameters']=config['parameters']

    if not suppress_output:
        if type(cfg['scan_parameter']) is list:
            for param,values in zip(cfg['scan_parameter'],cfg['scan_values']):
                print('Will scan parameter '+param)
                print('Will scan value '+str(values))
        else:
            print('Will scan parameter '+cfg['scan_parameter'])
            print('Will scan value '+str(cfg['scan_values'][0]))

    return cfg

def make_kube_yaml(mount_point,input_file,bucket,output_dir,npcs,num_iter,var_name,save_every,
                   cross_validate,model_progress,whiten,save_model,restarts,worker_dicts,
                   other_parameters,ext,job_name,image,ncpus,restart_policy):

    bash_commands=['/bin/bash','-c']
    bash_arguments='kinect_model learn_model '+os.path.join(mount_point,input_file)
    gcs_options='-o allow_other --file-mode=777 --dir-mode=777'
    mount_arguments='mkdir '+mount_point+'; gcsfuse '+gcs_options+' '+bucket+' '+mount_point
    dir_arguments='mkdir -p '+output_dir
    param_commands=('--npcs '+str(npcs)+
                    ' --num-iter '+str(num_iter)+
                    ' --var-name '+var_name+
                    ' --save-every '+str(save_every))

    bash_commands=[yaml.scalarstring.DoubleQuotedScalarString(cmd) for cmd in bash_commands]

    if cross_validate:
        param_commands=param_commands+' --cross-validate'

    if model_progress:
        param_commands=param_commands+' --model-progress'

    if whiten:
        param_commands=param_commands+' --whiten'

    if save_model:
        param_commands=param_commands+' --save-model'

    # TODO cross-validation

    if restarts>1:
        worker_dicts=[val for val in worker_dicts for _ in xrange(restarts)]

    njobs=len(worker_dicts)
    job_dict=[{'apiVersion':'v1','kind':'Pod'}]*njobs

    yaml_string=''

    for itr,job in enumerate(worker_dicts):

        # need some unique stuff to specify what this job is, do some good bookkeeping for once

        job_dict[itr]['metadata'] = {'name':job_name+'-{:d}'.format(itr),
            'labels':{'jobgroup':job_name}}

        # scan parameters are commands, along with any other specified parameters
        # build up the list for what we're going to pass to the command line

        all_parameters=merge_dicts(other_parameters,worker_dicts[itr])

        output_dir_string=os.path.join(output_dir,'job_{:06d}{}'.format(itr,ext))
        issue_command=mount_arguments+'; '+dir_arguments+'; '+bash_arguments+' '+output_dir_string
        issue_command=issue_command+' '+param_commands

        for param,value in all_parameters.iteritems():
            param_name=yaml.scalarstring.DoubleQuotedScalarString('--'+param)
            param_value=yaml.scalarstring.DoubleQuotedScalarString(str(value))
            issue_command=issue_command+' '+param_name
            issue_command=issue_command+' '+param_value

        # TODO: cross-validation

        job_dict[itr]['spec'] = {'containers':[{'name':'kinect-modeling','image':image,'command':bash_commands,
            'args':[yaml.scalarstring.DoubleQuotedScalarString(issue_command)],
            'securityContext':{'privileged': True},
            'resources':{'requests':{'cpu': '{:d}m'.format(int(ncpus*.6*1e3)) }}}],'restartPolicy':restart_policy}

        yaml_string='{}\n{}\n---'.format(yaml_string,yaml.dump(job_dict[itr],Dumper=yaml.RoundTripDumper))

    return yaml_string

def kube_info(cluster_name):

    cluster_info={}

    try:
        test=subprocess.check_output(["gcloud", "container", "clusters", "describe", cluster_name])
    except subprocess.CalledProcessError, e:
        print "Error trying to call gcloud:\n", e.output

    try:
        images=subprocess.check_output("gcloud beta container images list | awk '{if(NR>1)print}'",shell=True).split('\n')
    except subprocess.CalledProcessError, e:
        print "Error trying to call gcloud:\n", e.output

    parsed_output=yaml.load(test, Loader=yaml.Loader)

    machine=parsed_output['nodeConfig']['machineType']
    re_machine=re.split('\-',machine)

    if re_machine[0]=='custom':
        cluster_info['ncpus']=int(re_machine[1])
    else:
        cluster_info['ncpus']=int(re_machine[2])

    cluster_info['cluster_name']=parsed_output['name']
    del images[-1]

    cluster_info['images']=images

    return cluster_info


# credit to http://stackoverflow.com/questions/14000893/specifying-styles-for-portions-of-a-pyyaml-dump
class blockseq( dict ): pass
def blockseq_rep(dumper, data):
    return dumper.represent_mapping( u'tag:yaml.org,2002:map', data, flow_style=False )

class flowmap( dict ): pass
def flowmap_rep(dumper, data):
    return dumper.represent_mapping( u'tag:yaml.org,2002:map', data, flow_style=True )


# from http://stackoverflow.com/questions/16782112/can-pyyaml-dump-dict-items-in-non-alphabetical-order
def represent_ordereddict(dumper, data):
    value = []

    for item_key, item_value in data.items():
        node_key = dumper.represent_data(item_key)
        node_value = dumper.represent_data(item_value)

        value.append((node_key, node_value))

    return yaml.nodes.MappingNode(u'tag:yaml.org,2002:map', value)
