from __future__ import division
import click
import os.path
import datetime
from tqdm import tqdm
from train.models import ARHMM
import ruamel.yaml as yaml
from collections import OrderedDict
from train.util import merge_dicts, train_model, whiten_all
from util import enum, save_dict, load_pcs, read_cli_config, copy_model, get_parameters_from_model
from mpi4py import MPI

# leave the user with the option to use (A) MPI

@click.group()
def cli():
    pass

@cli.command()
@click.argument("param_file", type=click.Path(exists=True))
@click.argument("input_file", type=click.Path(exists=True))
@click.argument("dest_file", type=click.Path(dir_okay=True,writable=True))
@click.option("--cross-validate","-c",is_flag=True)
@click.option("--num-iter", "-n", type=int, default=100)
@click.option("--restarts", "-r", type=int, default=1)
@click.option("--var_name", type=str, default='features')
@click.option("--save-every", "-s", type=int, default=1)
@click.option("--save-model","-m", is_flag=True)
@click.option("--model-progress","-p",is_flag=True)
@click.option("--npcs", type=int, default=10)
def parameter_scan_mpi(param_file, input_file, dest_file, cross_validate,
    num_iter, restarts, var_name, save_every, save_model, model_progress,npcs):

    tags = enum('READY','DONE','EXIT','START')

    comm = MPI.COMM_WORLD
    size  = comm.Get_size()
    rank = comm.Get_rank()
    status = MPI.Status()

    # node w/ rank 0 sends and saves data

    data_dict={}

    if rank==0:

        save_dict(filename=dest_file,print_message=True)
        [scan_dicts,scan_parameter,scan_values,other_parameters,scan_settings]=read_cli_config(param_file)

        # get them pc's

        data_dict=load_pcs(filename=input_file, var_name=var_name, npcs=npcs)

        # use a list of dicts, with everything formatted ready to go

        click.echo('Whitening the training data')
        data_dict=whiten_all(data_dict)

        all_keys=data_dict.keys()
        worker_dicts=[]
        nframes=[tmp.shape[0] for key,tmp in data_dict.iteritems()]

        # pre-initialize everything

        # set up the tasks as a $$$$-ton of dictionaries specifying the parameters in the sweep
        # for cross validation need to specify the fold as well

        if cross_validate:
            for cv_idx, test_key in enumerate(all_keys):
                train_keys = [key for key in all_keys if key not in test_key]
                for scan_idx, use_dict in enumerate(scan_dicts):
                    for restart_idx in xrange(restarts):
                        worker_dicts.append({'scan_dict': use_dict,
                                            'index': (restart_idx,cv_idx,scan_idx),
                                            'train_keys': train_keys,
                                            'test_key': test_key,
                                            'other_parameters':other_parameters})
        else:
            for scan_idx, use_dict in enumerate(scan_dicts):
                for restart_idx in xrange(restarts):
                    worker_dicts.append({'scan_dict': use_dict,
                        'index': (restart_idx, scan_idx),
                        'other_parameters': other_parameters})


        # everything is a list in the end

        labels = [None] * len(worker_dicts)
        loglikes = [None] * len(worker_dicts)
        heldout_ll = None
        models = None

        if cross_validate:
            heldout_ll = [None] * len(worker_dicts)

        if save_model:
            models = [None] * len(worker_dicst)

        # each worker gets a dictionary, the tuple index points to where the data will end up

    data_dict = comm.bcast(data_dict,root=0)

    if rank == 0:

        click.echo('Starting training...')

        task_index = 0
        num_workers = size - 1
        closed_workers = 0
        pbar = tqdm(total=len(worker_dicts),smoothing=0)

        while closed_workers < num_workers:

            # take 'em as they're available, farm out the job

            data = None
            data = comm.recv(source=MPI.ANY_SOURCE, tag=MPI.ANY_TAG, status=status)

            source = status.Get_source()
            tag = status.Get_tag()

            if tag == tags.READY:

                if task_index < len(worker_dicts):
                    worker_dicts[task_index]['task_index']=task_index
                    comm.send(worker_dicts[task_index], dest=source, tag=tags.START)
                    task_index += 1
                else:
                    comm.send(None, dest=source, tag=tags.EXIT)

            elif tag == tags.DONE:

                # these lists are getting big, may want to resort to pickle dumping and loading in a for
                # loop
                # sort out the data brutha

                worker_index = data['index']
                worker_task_index = data['task_index']

                labels[worker_index]=data['labels']
                loglikes[worker_index]=data['loglikes']

                if cross_validate:
                    heldout_ll[worker_index]=data['heldout_ll']

                pbar.update(1)

            elif tag == tags.EXIT:

                closed_workers += 1
        pbar.close()

        click.echo('Saving results to '+dest_file)
        export_dict=dict({'loglikes':loglikes, 'labels':labels, 'heldout_ll':heldout_ll,
                          'scan_dicts':scan_dicts},**scan_settings)
        save_dict(filename=dest_file,obj_to_save=export_dict)

    else:

        while True:

            arhmm = None
            loglikes = None
            labels = None
            worker_dict = None

            comm.send(None, dest=0, tag=tags.READY)
            worker_dict = comm.recv(source=0, tag=MPI.ANY_TAG, status=status)
            tag = status.Get_tag()

            if tag == tags.START:


                # if we get marching orders, fire up the task

                tmp_parameters=merge_dicts(worker_dict['other_parameters'],worker_dict['scan_dict'])

                # TODO: add model saving

                if not cross_validate:
                    tmp_parameters = merge_dicts(worker_dict['other_parameters'],
                                                worker_dict['scan_dict'])
                    arhmm=ARHMM(data_dict=data_dict, **tmp_parameters)
                    [_,loglikes,labels]=train_model(model=arhmm,
                                                    num_iter=num_iter,
                                                    save_every=save_every,
                                                    cli=True,
                                                    position=rank,
                                                    disable=not model_progress)
                    comm.send({'index':worker_dict['index'],
                        'loglikes':loglikes,'labels':labels}, dest=0, tag=tags.DONE)
                else:
                    train_data=OrderedDict((i,data_dict[i]) for i in worker_dict['train_keys'])
                    arhmm=ARHMM(data_dict=train_data, **tmp_parameters)
                    [arhmm,loglikes,labels]=train_model(model=arhmm,
                                                        num_iter=num_iter,
                                                        cli=True,
                                                        position=rank,
                                                        disable=not model_progress)

                    heldout_ll = arhmm.log_likelihood(data_dict[worker_dict['test_key']])
                    comm.send({'index':worker_dict['index'],'heldout_ll':heldout_ll}, dest=0, tag=tags.DONE)

            elif tag == tags.EXIT:

                break

        comm.send(None, dest=0, tag=tags.EXIT)

# this will take some parameter scan specification and create a yaml file we can pipe into kubectl
@cli.command()
@click.argument("param_file", type=click.Path(exists=True))
@click.option("--cross-validate","-c",is_flag=True)
@click.option("--num-iter", "-n", type=int, default=100)
@click.option("--restarts", "-r", type=int, default=1)
@click.option("--var_name", type=str, default='features')
@click.option("--save-every", "-s", type=int, default=1)
@click.option("--save-model","-m", is_flag=True)
@click.option("--model-progress","-p",is_flag=True)
@click.option("--npcs", type=int, default=10)
@click.option("--whiten","-w", type=bool, default=True)
@click.option("--image","-i",type=str, default="kinect-modeling")
@click.option("--job-name", type=str, default="kubejob")
@click.option("--submit-job", is_flag=True)
@click.option("--output-dir", type=str, default="")
@click.option("--ext","-e",type=str, default=".p.z")
@click.option("--mount-point", type=str, default="/mnt/modeling_bucket")
@click.option("--bucket","-b",type=str, default="modeling-bucket")
@click.option("--restart-policy", type=str, default="Never")
@click.option("--ncpus", type=int, default=4)
def parameter_scan_kube(param_file, cross_validate,
    num_iter, restarts, var_name, save_every, save_model, model_progress,npcs,whiten,
    image, job_name, output_dir, ext, submit_job, mount_point, bucket, restart_policy, ncpus):

    # use pyyaml to build up a list of worker dictionaries, make a giant yaml
    # file that we can then farm out to Kubernetes cluster using kubectl

    cfg=read_cli_config(param_file,suppress_output=True)

    if 'npcs' in cfg:
        npcs=cfg['npcs']

    if 'num-iter' in cfg:
        num_iter=cfg['num-iter']

    njobs=len(cfg['worker_dicts'])
    job_dict=[{'apiVersion':'v1','kind':'Pod'}]*njobs

    if not output_dir:
        output_dir=job_name+'_{:%Y-%m-%d_%H-%M-%S}'.format(datetime.datetime.now())

    output_dir=os.path.join(mount_point,output_dir)

    bash_commands=['/bin/bash','-c']
    bash_arguments='kinect_model learn_model '+os.path.join(mount_point,cfg['input_file'])
    gcs_options='-o allow_other --file-mode=777 --dir-mode=777'
    mount_arguments='mkdir '+mount_point+'; gcsfuse '+gcs_options+' '+bucket+' '+mount_point
    dir_arguments='mkdir -p '+output_dir
    param_commands=('--npcs '+str(npcs)+' --save-every '+str(save_every)+
                    ' --num-iter '+str(num_iter)+
                    ' --restarts '+str(restarts)+
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


    # TODO: repeats and cross-validation
    # TODO: pull in a notes field!
    # TODO: specify nodes as well?

    for itr,job in enumerate(cfg['worker_dicts']):

        # need some unique stuff to specify what this job is, do some good bookkeeping for once

        job_dict[itr]['metadata'] = {'name':job_name+'-{:d}'.format(itr),
            'labels':{'jobgroup':job_name}}

        # scan parameters are commands, along with any other specified parameters
        # build up the list for what we're going to pass to the command line

        all_parameters=merge_dicts(cfg['other_parameters'],cfg['worker_dicts'][itr])

        output_dir_string=os.path.join(output_dir,'job_{:06d}{}'.format(itr,ext))
        issue_command=mount_arguments+'; '+dir_arguments+'; '+bash_arguments+' '+output_dir_string
        issue_command=issue_command+' '+param_commands

        for param,value in all_parameters.iteritems():
            param_name=yaml.scalarstring.DoubleQuotedScalarString('--'+param)
            param_value=yaml.scalarstring.DoubleQuotedScalarString(str(value))
            issue_command=issue_command+' '+param_name
            issue_command=issue_command+' '+param_value

        # TODO: cross-validation

        job_dict[itr]['spec'] = {'containers':[{'name':'test','image':image,'command':bash_commands,
            'args':[yaml.scalarstring.DoubleQuotedScalarString(issue_command)],
            'securityContext':{'privileged': True},'resources':{'requests':{'cpu': '{:d}m'.format(int(ncpus*.6*1e3)) }}}],'restartPolicy':restart_policy}

        print(yaml.dump(job_dict[itr],Dumper=yaml.RoundTripDumper))
        print('---')

        if submit_job:
            # either we can dump into a file and deal with kubectl on our own
            # or we can simply pipe the yaml output to kubectl

            raise NotImplementedError

# this is the entry point for learning models over Kubernetes, expose all
# parameters we could/would possibly scan over
@cli.command()
@click.argument("input_file", type=click.Path(exists=True))
@click.argument("dest_file", type=click.Path(dir_okay=True,writable=True))
@click.option("--hold-out","-h", type=int, default=-1)
@click.option("--num-iter", "-n", type=int, default=100)
@click.option("--restarts","-r",type=int,default=1)
@click.option("--var-name", type=str, default='features')
@click.option("--save-every", "-s", type=int, default=1)
@click.option("--save-model","-m", is_flag=True)
@click.option("--model-progress","-p", is_flag=True)
@click.option("--npcs", type=int, default=10)
@click.option("--whiten", is_flag=True)
@click.option("--kappa","-k", type=float, default=1e8)
@click.option("--gamma","-g",type=float, default=1e3)
@click.option("--nlags",type=int, default=3)
def learn_model(input_file, dest_file, hold_out, num_iter, restarts, var_name, save_every,
    save_model, model_progress, npcs, whiten, kappa, gamma, nlags):

    data_dict=load_pcs(filename=input_file, var_name=var_name, npcs=npcs)

    # use a list of dicts, with everything formatted ready to go

    model_parameters={
        'gamma':gamma,
        'kappa':kappa,
        'nlags':nlags
    }

    if whiten:

        click.echo('Whitening the training data')
        data_dict=whiten_all(data_dict)

    all_keys=data_dict.keys()

    if hold_out>=0:
        train_data=OrderedDict((i,data_dict[i]) for i in all_keys if i not in all_keys[hold_out])
        test_data=data_dict[all_keys[hold_out]]
    else:
        train_data=data_dict

    arhmm=ARHMM(data_dict=train_data, **model_parameters)
    [arhmm,loglikes,labels]=train_model(model=arhmm,
                                    num_iter=num_iter,
                                    cli=True,
                                    disable=not model_progress)

    if hold_out>=0:
        heldout_ll=arhmm.log_likelihood(data_dict[all_keys[hold_out]])
    else:
        heldout_ll=None

    export_dict=dict({'loglikes':loglikes, 'labels':labels, 'heldout_ll':heldout_ll,
                      'model_parameters':get_parameters_from_model(arhmm)})
    save_dict(filename=dest_file,obj_to_save=export_dict)


# @cli.command()
# @click.argument("input_file", type=click.Path(exists=True))
# @click.argument("dest_file", type=click.Path(dir_okay=True,writable=True))
# @click.argument("--each","-e", is_flag=True)
# @click.argument("--var_name","-v",type=str, default="features")
# def whiten(input_file,dest_file,each,var_name):
#     pass
