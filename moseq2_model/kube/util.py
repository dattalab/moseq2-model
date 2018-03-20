import ruamel.yaml as yaml
import os
import re
import subprocess
import tempfile
import shutil
from moseq2_model.util import merge_dicts, load_pcs
from sys import platform
from copy import deepcopy


# wow how did you get so parameters
def make_kube_yaml(mount_point, input_file, bucket, output_dir, npcs, num_iter,
                   var_name, save_every, cross_validate, model_progress, whiten,
                   save_model, restarts, worker_dicts, other_parameters, ext,
                   job_name, image, ncpus, restart_policy, gcs_options, robust,
                   suffix, kind, separate_trans, nmem, hold_out,
                   nfolds=None, start_num=None, **kwargs):

    # TODO: better safeguards against user stupidity

    bucket_dir = os.path.join(bucket, output_dir, job_name+suffix)
    output_dir = os.path.join(mount_point, output_dir, job_name+suffix)

    bash_commands = ['/bin/bash', '-c']
    bash_arguments = 'moseq2-model learn-model '+os.path.join(mount_point, input_file)
    mount_arguments = 'mkdir '+mount_point+'; gcsfuse '+gcs_options+' '+bucket+' '+mount_point
    dir_arguments = 'mkdir -p '+output_dir
    param_commands = ('--npcs '+str(npcs) +
                      ' --num-iter '+str(num_iter) +
                      ' --var-name '+var_name +
                      ' --save-every '+str(save_every) +
                      ' --whiten '+whiten)

    # if we're using ssh need a whole new ****-load of parameters

    bash_commands = [yaml.scalarstring.DoubleQuotedScalarString(cmd) for cmd in bash_commands]

    if model_progress:
        param_commands = param_commands+' --model-progress'

    if save_model:
        param_commands = param_commands+' --save-model'

    if separate_trans:
        param_commands = param_commands+' --separate-trans'

    if robust:
        param_commands = param_commands+' --robust'

    if hold_out and nfolds:
        new_dicts = []
        param_commands = param_commands+' --hold-out'
        # split number of keys into nfolds equally

        for i in range(len(worker_dicts)):
            worker_dicts[i]['nfolds'] = nfolds
            worker_dicts[i]['hold-out-seed'] = 1
            new_dicts.append(worker_dicts[i].copy())

        worker_dicts = new_dicts

    # allow for internal loop restarts too?

    if restarts > 1:
        new_dicts = []
        for i in range(len(worker_dicts)):
            for j in range(restarts):
                worker_dicts[i]['restart'] = j
                new_dicts.append(worker_dicts[i].copy())

        worker_dicts = new_dicts

    # worker_dicts=[val for val in worker_dicts for _ in range(restarts)]

    output_dicts = deepcopy(worker_dicts)
    njobs = len(worker_dicts)

    if kind == 'Pod':
        job_dict = [{'apiVersion': 'v1', 'kind': 'Pod'}]*njobs
    elif kind == 'Job':
        job_dict = [{'apiVersion': 'batch/v1', 'kind': 'Job'}]*njobs

    yaml_string = ''

    for itr, job in enumerate(worker_dicts):

        # need some unique stuff to specify what this job is, do some good bookkeeping for once

        job_dict[itr]['metadata'] = {
            'name': job_name+'-{:d}'.format(itr+start_num),
            'labels': {'jobgroup': job_name}
            }

        # scan parameters are commands, along with any other specified parameters
        # build up the list for what we're going to pass to the command line

        worker_dicts[itr].pop('restart', 0)
        all_parameters = merge_dicts(other_parameters, worker_dicts[itr])

        output_dir_string = os.path.join(output_dir, 'job_{:06d}{}'.format(itr, ext))
        issue_command = mount_arguments+'; '+dir_arguments+'; '+bash_arguments+' '+output_dir_string
        issue_command = issue_command+' '+param_commands

        for param, value in all_parameters.items():
            param_name = yaml.scalarstring.DoubleQuotedScalarString('--'+param)
            param_value = yaml.scalarstring.DoubleQuotedScalarString(str(value))
            issue_command = issue_command+' '+param_name
            issue_command = issue_command+' '+param_value

        # TODO: cross-validation

        container_dict = {
            'containers': [
                {'name': 'moseq2-modeling',
                 'image': image,
                 'command': bash_commands,
                 'args': [yaml.scalarstring.DoubleQuotedScalarString(issue_command)],
                 'securityContext': {'privileged': True},
                 'resources': {
                     'requests': {'cpu': '{:d}m'.format(int(ncpus*.9*1e3)),
                                  'memory': '{:d}Mi'.format(int(nmem))}}}],
            'restartPolicy': restart_policy}

        if kind == 'Pod':
            job_dict[itr]['spec'] = container_dict
        elif kind == 'Job':
            job_dict[itr]['spec'] = {
                'template': {
                    'metadata': {
                        'name': job_name
                        },
                    'spec': container_dict}}

        output_dicts[itr]['filename'] = output_dir_string

        yaml_string = '{}\n{}\n---'.format(yaml_string, yaml.dump(job_dict[itr], Dumper=yaml.RoundTripDumper))

    return yaml_string, output_dicts, output_dir, bucket_dir


def kube_cluster_check(cluster_name, ncpus, image, preflight=False):

    cluster_info = {}

    try:
        test = subprocess.check_output(["gcloud", "container", "clusters", "describe", cluster_name])
    except ValueError as e:
        print("Error trying to call gcloud: {}\n".format(e.output))

    try:
        images = subprocess.check_output("gcloud beta container images list | awk '{if(NR>1)print}'",
                                         shell=True).split('\n')
    except ValueError as e:
        print("Error trying to call gcloud: {}\n".format(e.output))

    parsed_output = yaml.load(test, Loader=yaml.Loader)
    machine = parsed_output['nodeConfig']['machineType']
    re_machine = re.split('-', machine)

    if re_machine[0] == 'custom':
        cluster_info['ncpus'] = int(re_machine[1])
    else:
        cluster_info['ncpus'] = int(re_machine[2])

    cluster_info['cluster_name'] = parsed_output['name']
    cluster_info['scopes'] = parsed_output['nodeConfig']['oauthScopes']
    del images[-1]

    cluster_info['images'] = images

    preflight_check(flag=ncpus*.9 <= cluster_info['ncpus']*.9, preflight=preflight,
                    msg='NCPUS',
                    err_msg="User setting ncpus {:d} more than 90% number of cpus in cluster {:d}".format(
                        ncpus, cluster_info['ncpus']))

    preflight_check(flag=image in cluster_info['images'], preflight=preflight,
                    msg='Docker image',
                    err_msg="User-defined image {} not available, available images are {}".format(
                        image, cluster_info['images']))

    preflight_check(flag='https://www.googleapis.com/auth/devstorage.full_control' in cluster_info['scopes'],
                    preflight=preflight,
                    msg="Cluster scope",
                    err_msg="Scope storage-full not found in current cluster {}".format(cluster_name))

    return cluster_info


def kube_check_mount(bucket, gcs_options="", input_file=None, ssh_key=None, ssh_user=None,
                     ssh_remote_server=None, ssh_remote_dir=None, preflight=False,
                     var_name=None, npcs=None, **kwargs):

    # TODO: test for existence of input file
    # TODO: clean this shit up

    PASS = True
    data_len = None

    try:

        gcs_tmp = os.path.join(os.path.expanduser("~"), 'gcs_test')
        if not os.path.isdir(gcs_tmp):
            os.mkdir(gcs_tmp)

        try:
            if gcs_options:
                test_gcs_mount = subprocess.check_output(
                    "gcsfuse "+gcs_options+' '+bucket+' '+gcs_tmp, shell=True, stderr=subprocess.STDOUT)
            else:
                test_gcs_mount = subprocess.check_output(
                    "gcsfuse "+bucket+' '+gcs_tmp, shell=True, stderr=subprocess.STDOUT)
        except ValueError as e:
            print("Error when mounting gcs bucket {}:\n".format(e.output))

        PASS = preflight_check(flag=os.access(gcs_tmp, os.W_OK | os.X_OK), preflight=preflight,
                               msg='GCS Bucket access',
                               err_msg="GCS bucket is not writeable, look at gcs_options")

        try:
            data_dict = load_pcs(filename=os.path.join(gcs_tmp, input_file), var_name=var_name, npcs=npcs)
            data_len = len(data_dict)
        except ValueError as e:
            print("Error when loading data {}:\n".format(e.output))

    except Exception as e:
        print(str(e))

    finally:

        try:
            if platform == 'linux' or platform == 'linux2':
                test_gcs_umount = subprocess.check_output(["fusermount", "-uz", gcs_tmp])
            elif platform == 'darwin':
                test_gcs_umount = subprocess.check_output(["umount", gcs_tmp])
        except ValueError as e:
            print("Error when unmounting gcs bucket {}:\n".format(e.output))

    if preflight and PASS:
        print('ALL SYSTEMS GO')

    return PASS, data_len


def make_temporary_copy(path):

    tmp_dir = tempfile.mkdtemp()
    use_file = os.path.join(tmp_dir, 'tmp_file')
    shutil.copy2(path, use_file)

    return use_file


def preflight_check(flag, preflight, msg='Variable check', err_msg='Error'):

    chk = True

    if not flag and not preflight:
        raise ValueError(err_msg)
    elif not flag:
        print(msg+'...FAIL')
        chk = False
    elif preflight:
        print(msg+'...PASS')
    else:
        pass

    return chk
