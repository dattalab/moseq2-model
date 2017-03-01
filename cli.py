from __future__ import division
import click
import os.path
from tqdm import tqdm
from train.models import ARHMM
import ruamel.yaml as yaml
from collections import OrderedDict
from train.util import merge_dicts, train_model, whiten_all
from util import enum, save_dict, load_pcs, read_cli_config, copy_model, flowmap, blockseq, flowmap_rep, blockseq_rep
from mpi4py import MPI

# leave the user with the option to use (A) MPI

@click.group()
def cli():
    pass

@cli.command()
@click.argument("paramfile", type=click.Path(exists=True))
@click.argument("inputfile", type=click.Path(exists=True))
@click.argument("destfile", type=click.Path(dir_okay=True,writable=True))
@click.option("--cross-validate","-c",is_flag=True)
@click.option("--num-iter", "-n", type=int, default=100)
@click.option("--restarts", "-r", type=int, default=1)
@click.option("--varname", type=str, default='features')
@click.option("--save-every", "-s", type=int, default=1)
@click.option("--save-model","-m", is_flag=True)
@click.option("--model-progress","-p",is_flag=True)
@click.option("--npcs", type=int, default=10)
def parameter_scan_mpi(paramfile, inputfile, destfile, cross_validate,
    num_iter, restarts, varname, save_every, save_model, model_progress,npcs):

    tags = enum('READY','DONE','EXIT','START')

    comm = MPI.COMM_WORLD
    size  = comm.Get_size()
    rank = comm.Get_rank()
    status = MPI.Status()

    # node w/ rank 0 sends and saves data

    data_dict={}

    if rank==0:

        save_dict(filename=destfile,print_message=True)
        [scan_dicts,scan_parameter,scan_values,other_parameters,scan_settings]=read_cli_config(paramfile)

        # get them pc's

        data_dict=load_pcs(filename=inputfile, varname=varname, npcs=npcs)

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

        click.echo('Saving results to '+destfile)
        export_dict=dict({'loglikes':loglikes, 'labels':labels, 'heldout_ll':heldout_ll,
                          'scan_dicts':scan_dicts},**scan_settings)
        save_dict(filename=destfile,obj_to_save=export_dict)

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
@click.argument("paramfile", type=click.Path(exists=True))
@click.argument("destfile", type=click.Path(dir_okay=True,writable=True))
@click.option("--cross-validate","-c",is_flag=True)
@click.option("--num-iter", "-n", type=int, default=100)
@click.option("--restarts", "-r", type=int, default=1)
@click.option("--varname", type=str, default='features')
@click.option("--save-every", "-s", type=int, default=1)
@click.option("--save-model","-m", is_flag=True)
@click.option("--model-progress","-p",is_flag=True)
@click.option("--npcs", type=int, default=10)
@click.option("--image","-i",type=str, default="kinect_modeling")
@click.option("--job-name", type=str, default="kubejob")
@click.option("--submit-job", is_flag=True)
@click.option("--restart-policy", type=str, default="Never")
def parameter_scan_kube(paramfile, destfile, cross_validate,
    num_iter, restarts, varname, save_every, save_model, model_progress,npcs,
    image, job_name, submit_job,restart_policy):

    # use pyyaml to build up a list of worker dictionaries, make a giant yaml
    # file that we can then farm out to Kubernetes cluster using kubectl

    cfg=read_cli_config(paramfile)
    njobs=len(cfg['worker_dicts'])
    job_dict=[{'apiVersion':'batch/v1','kind':'Job'}]*njobs

    bash_commands=['sh','-c','kinect_model','learn_model']

    for itr,job in enumerate(cfg['worker_dicts']):

        # need some unique stuff to specify what this job is, do some good bookkeeping for once

        job_dict[itr]['metadata'] = {'name':job_name+'-'+str(itr),
            'labels':{'jobgroup':job_name}}

        # scan parameters are commands, along with any other specified parameters
        # build up the list for what we're going to pass to the command line

        all_parameters=merge_dicts(cfg['other_parameters'],cfg['worker_dicts'][itr])
        issue_command=[yaml.scalarstring.DoubleQuotedScalarString(cmd) for cmd in bash_commands]

        for param,value in all_parameters.iteritems():
            param_name=yaml.scalarstring.DoubleQuotedScalarString('--'+param)
            param_value=yaml.scalarstring.DoubleQuotedScalarString(str(value))
            issue_command.append(param_name)
            issue_command.append(param_value)

        # TODO: cross-validation

        job_dict[itr]['spec'] = {'template':
            {'metadata':{'name':'example','labels':{'jobgroup':'example'}},
            'spec':{'containers':[{'name':'test','image':'busybox',
            'command':issue_command}]},
            'restartPolicy':restart_policy}}

        print(yaml.dump(job_dict[itr],Dumper=yaml.RoundTripDumper))
        print('---')

        if submit_job:
            # either we can dump into a file and deal with kubectl on our own
            # or we can simply pipe the yaml output to kubectl

            raise NotImplementedError

    #print(yaml.dump_all(job_dict,Dumper=yaml.RoundTripDumper))

    # metadata (job name etc.)

    # need a storage directory for each job too (store in Docker and retrieve or something?)

    # automagically call kubectl if the user wants to?

# this is the entry point for learning models over Kubernetes, expose all
# parameters we could/would possibly scan over
@cli.command()
@click.argument("inputfile", type=click.Path(exists=True))
@click.argument("destfile", type=click.Path(dir_okay=True,writable=True))
@click.option("--hold-out","-h", type=int, default=-1)
@click.option("--num-iter", "-n", type=int, default=100)
@click.option("--varname", type=str, default='features')
@click.option("--save-every", "-s", type=int, default=1)
@click.option("--save-model","-m", is_flag=True)
@click.option("--model-progress","-p", is_flag=True)
@click.option("--npcs", type=int, default=10)
@click.option("--kappa","-k", type=float, default=1e8)
@click.option("--gamma","-g",type=float, default=1e3)
def learn_model(inputfile, destfile, cross_validate, num_iter, varname, save_every,
    save_model, model_progress, npcs, kappa, gamma):



    pass


# whiten the data, either append file or save to a new file

# @cli.command()
# @click.argument("inputfile", type=click.Path(exists=True))
# @click.argument("destfile", type=click.Path(dir_okay=True,writable=True))
# @click.argument("--each","-e", is_flag=True)
# @click.argument("--varname","-v",type=str, default="features")
# def whiten(inputfile,destfile,each,varname):
#     pass
