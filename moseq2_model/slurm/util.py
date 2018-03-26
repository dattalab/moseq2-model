import os
from copy import deepcopy


# wow how did you get so parameters
def make_slurm_batch(mount_point, input_file, bucket, output_dir,
                     restarts, worker_dicts, ext,
                     job_name, image, ncpus, restart_policy, gcs_options,
                     suffix, kind, nmem, prefix='', start_num=None, parameters={}, flags={},
                     **kwargs):

    # TODO: better safeguards against user stupidity

    bucket_dir = os.path.join(bucket, output_dir, job_name+suffix)
    output_dir = os.path.join(mount_point, output_dir, job_name+suffix)

    bash_arguments = 'moseq2-model learn-model '+os.path.join(mount_point, input_file)
    mount_arguments = 'mkdir '+mount_point+'; gcsfuse '+gcs_options+' '+bucket+' '+mount_point
    dir_arguments = 'mkdir -p '+output_dir

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    if not os.access(output_dir, os.W_OK):
        raise IOError('Output directory is not writable.')

    param_commands = ''
    for k, v in parameters.items():
        param_commands += ' --{} {}'.format(k, str(v))

    for k, v in flags.items():
        if v:
            param_commands += ' --{}'.format(k)

    # if we're using ssh need a whole new ****-load of parameters

    # allow for internal loop restarts too?

    if restarts > 1:
        new_dicts = []
        for i in range(len(worker_dicts)):
            for j in range(restarts):
                worker_dicts[i]['restart'] = j
                new_dicts.append(worker_dicts[i].copy())

        worker_dicts = new_dicts

    # worker_dicts=[val for val in worker_dicts for _ in range(restarts)]
    # make sure we dump via yaml to the output directory!

    njobs = len(worker_dicts)

    if kind == 'Pod':
        job_dict = [{'apiVersion': 'v1', 'kind': 'Pod'}]*njobs
    elif kind == 'Job':
        job_dict = [{'apiVersion': 'batch/v1', 'kind': 'Job'}]*njobs

    for itr, job in enumerate(worker_dicts):

        # need some unique stuff to specify what this job is, do some good bookkeeping for once

        job_dict[itr]['metadata'] = {
            'name': job_name+'-{:d}'.format(itr+start_num),
            'labels': {'jobgroup': job_name}
            }

        # scan parameters are commands, along with any other specified parameters
        # build up the list for what we're going to pass to the command line

        worker_dicts[itr].pop('restart', 0)
        # all_parameters = merge_dicts(other_parameters, worker_dicts[itr])

        output_dir_string = os.path.join(output_dir, 'job_{:06d}{}'.format(itr, ext))

        issue_command = 'sbatch --ntasks=1 --cpus-per-task={:d} --mem={:d} --wrap "'.format(ncpus, nmem)

        if prefix:
            issue_command += prefix+'; '

        # if mount_point:
        #     issue_command += mount_arguments+'; '+dir_arguments+'; '+bash_arguments+' '+output_dir_string
        # else:
        #     issue_command += dir_arguments+'; '+bash_arguments+' '+output_dir_string

        issue_command = issue_command+' '+param_commands

        for param, value in worker_dicts[itr].items():
            issue_command = issue_command+' --{} {}'.format(param, str(value))

        issue_command += '"'

        # print the command then sleep to give slurm a break
        job_dict[itr]['filename'] = output_dir_string

        print(issue_command)
        print('sleep 0.3')

    return job_dict, output_dir, bucket_dir
