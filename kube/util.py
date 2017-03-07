from __future__ import division
import ruamel.yaml as yaml
import itertools
import os
import re
import subprocess
from collections import OrderedDict
from kinect_modeling.util import merge_dicts

def make_kube_yaml(mount_point,input_file,bucket,output_dir,npcs,num_iter,var_name,save_every,
                   cross_validate,model_progress,whiten,save_model,restarts,worker_dicts,
                   other_parameters,ext,job_name,image,ncpus,restart_policy,
                   ssh_key=None, ssh_user=None, ssh_remote_server=None,ssh_remote_dir=None, ssh_mount_point=None,**kwargs):

    use_ssh=False
    bash_commands=['/bin/bash','-c']
    bash_arguments='kinect_model learn_model '+os.path.join(mount_point,input_file)
    gcs_options='-o allow_other --file-mode=777 --dir-mode=777'
    mount_arguments='mkdir '+mount_point+'; gcsfuse '+gcs_options+' '+bucket+' '+mount_point
    dir_arguments='mkdir -p '+output_dir
    param_commands=('--npcs '+str(npcs)+
                    ' --num-iter '+str(num_iter)+
                    ' --var-name '+var_name+
                    ' --save-every '+str(save_every))

    # if we're using ssh need a whole new ****-load of parameters

    if ssh_key and ssh_user and ssh_remote_dir and ssh_mount_point and ssh_remote_server:
        mount_arguments=mount_arguments+'; '+kube_ssh_command(ssh_key=ssh_key,
                                                              ssh_user=ssh_user,
                                                              ssh_remote_dir=ssh_remote_dir,
                                                              ssh_remote_server=ssh_remote_server,
                                                              ssh_mount_point=ssh_mount_point)
        bash_arguments='kinect_model learn_model '+os.path.join(ssh_mount_point,input_file)

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
    except ValueError as e:
        print "Error trying to call gcloud:\n", e.output

    try:
        images=subprocess.check_output("gcloud beta container images list | awk '{if(NR>1)print}'",shell=True).split('\n')
    except ValueError as e:
        print "Error trying to call gcloud:\n", e.output

    parsed_output=yaml.load(test, Loader=yaml.Loader)
    machine=parsed_output['nodeConfig']['machineType']
    re_machine=re.split('\-',machine)

    if re_machine[0]=='custom':
        cluster_info['ncpus']=int(re_machine[1])
    else:
        cluster_info['ncpus']=int(re_machine[2])

    cluster_info['cluster_name']=parsed_output['name']
    cluster_info['scopes']=parsed_output['nodeConfig']['oauthScopes']
    del images[-1]

    cluster_info['images']=images

    return cluster_info

def kube_ssh_command(ssh_key=None, ssh_user=None, ssh_remote_server=None, ssh_remote_dir=None, ssh_mount_point=None):

    mount_ssh='mkdir ~/.ssh/; cp '+os.path.join(ssh_key,'id_rsa*')+' ~/.ssh/'+'; chmod 400 ~/.ssh/id_rsa*'
    mount_ssh=mount_ssh+'; mkdir '+ssh_mount_point+'; sshfs -o allow_other -o StrictHostKeyChecking=no '\
        +ssh_user+'@'+ssh_remote_server+':'+ssh_remote_dir+' '+ssh_mount_point

    return mount_ssh

def kube_check_mount(gcs_options,bucket,ssh_key=None,ssh_user=None,ssh_remote_server=None,ssh_remote_dir=None):
    
