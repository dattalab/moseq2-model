from __future__ import division
import numpy as np
import click
import yaml
from tqdm import tqdm
from train.models import ARHMM
from collections import OrderedDict
from train.util import merge_dicts, train_model, whiten_all
from util import enum, save_dict, load_pcs, read_cli_config, copy_model
from mpi4py import MPI
import warnings

@click.group()
def cli():
    pass

@cli.command()
@click.argument("paramfile", type=click.Path(exists=True))
@click.argument("inputfile", type=click.Path(exists=True))
@click.argument("destfile", type=click.Path(dir_okay=True,writable=True))
@click.option("--num-iter", "-n", type=int, default=100)
@click.option("--restarts", "-r", type=int, default=1)
@click.option('--varname', type=str, default='features')
@click.option('--save-every', "-s", type=int, default=1)
@click.option('--model-progress',"-p",is_flag=True)
def parameter_scan(paramfile, inputfile, destfile, num_iter, restarts, varname, save_every,model_progress):

    warnings.filterwarnings("ignore", category=np.VisibleDeprecationWarning)
    tags = enum('READY','DONE','EXIT','START')

    comm = MPI.COMM_WORLD
    size  = comm.Get_size()
    rank = comm.Get_rank()
    status = MPI.Status()

    # node w/ rank 0 sends and saves data

    data_dict={}

    if rank==0:

        if destfile.endswith('.mat'):
            click.echo('Will save to MAT-file: '+destfile)
        elif destfile.endswith('.z'):
            click.echo('Will save compressed pickle: '+destfile)
        elif destfile.endswith('.pkl') or destfile.endswith('.p'):
            click.echo('Will save pickle: '+destfile)
        elif destfile.endswith('.h5'):
            raise NotImplementedError
        else:
            raise Exception('Output file format not understood')

        [scan_dicts,scan_parameter,scan_values,other_parameters,scan_settings]=read_cli_config(paramfile)

        if type(scan_parameter) is list:
            for param,values in zip(scan_parameter,scan_values):
                click.echo('Will scan parameter '+param)
                click.echo('Will scan value '+str(values))
        else:
            click.echo('Will scan parameter '+scan_parameter)
            click.echo('Will scan value '+str(scan_values[0]))

        #click.echo('Scan values '+str(scan_values))

        # get them pc's

        data_dict=load_pcs(filename=inputfile, varname=varname, pcs=10)

        # use a list of dicts, with everything formatted ready to go

        click.echo('Whitening the training data')

        data_dict=whiten_all(data_dict)

        worker_dicts=[]
        nframes=[tmp.shape[0] for key,tmp in data_dict.iteritems()]

        # pre-initialize everything

        labels = np.empty((restarts, len(scan_dicts), len(data_dict)), dtype=object)
        loglikes = np.zeros((restarts, len(scan_dicts), np.floor(num_iter/save_every)), dtype=np.float64)

        for scan_idx, use_dict in enumerate(scan_dicts):
            for restart_idx in xrange(restarts):
                worker_dicts.append({'scan_dict': use_dict,
                    'index': (restart_idx, scan_idx),
                    'other_parameters': other_parameters})
                # for data_idx in xrange(len(data_dict)):
                #      labels[restart_idx,scan_idx,data_idx]=np.zeros((np.floor(num_iter/save_every),nframes[data_idx]),dtype=np.int16)

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
                    #click.echo('Distributing task '+str(task_index))
                    comm.send(worker_dicts[task_index], dest=source, tag=tags.START)
                    task_index += 1
                else:
                    comm.send(None, dest=source, tag=tags.EXIT)

            elif tag == tags.DONE:
                # sort out the data brutha

                worker_idx = data['index']

                if type(data['labels']) is float and np.isnan(data['labels']):
                    labels[worker_idx[0]][worker_idx[1]][:]=np.nan
                else:
                    for label_itr,label in enumerate(data['labels']):
                        labels[(worker_idx[0],worker_idx[1],label_itr)]=label

                loglikes[worker_idx[0],worker_idx[1],:] = data['loglikes']

                pbar.update(1)

            elif tag == tags.EXIT:

                closed_workers += 1


        pbar.close()

        if restarts>1:

            # yeah it's ugly, sue me

            labels_to_save=np.empty(labels.shape[1:],dtype=object)
            loglikes_to_save=np.zeros(loglikes.shape[1:],dtype=np.float64)
            max_loglikes=np.max(loglikes,axis=2)
            best_models=np.argmax(max_loglikes,axis=0)

            for i in xrange(len(labels_to_save)):
                loglikes_to_save[i,:]=loglikes[best_models[i],i,:]
                for j in xrange(len(labels_to_save[0])):
                    labels_to_save[i][j]=labels[best_models[i]][i][j]
        else:

            labels_to_save=np.squeeze(labels)
            loglikes_to_save=np.squeeze(loglikes)

        click.echo('Saving results to '+destfile)
        export_dict=dict({'loglikes':loglikes_to_save, 'labels':labels_to_save,
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

            elif tag == tags.EXIT:

                break

        comm.send(None, dest=0, tag=tags.EXIT)


@cli.command()
@click.argument("paramfile", type=click.Path(exists=True))
@click.argument("inputfile", type=click.Path(exists=True))
@click.argument("destfile", type=click.Path(dir_okay=True,writable=True))
@click.option("--num-iter", "-n", type=int, default=100)
@click.option("--restarts", "-r", type=int, default=1)
@click.option('--varname', type=str, default='features')
@click.option('--model-progress',"-p",is_flag=True)
def cv_parameter_scan(paramfile, inputfile, destfile, num_iter, restarts, varname, model_progress):

    warnings.filterwarnings("ignore", category=np.VisibleDeprecationWarning)
    tags = enum('READY','DONE','EXIT','START')

    comm = MPI.COMM_WORLD
    size  = comm.Get_size()
    rank = comm.Get_rank()
    status = MPI.Status()

    # node w/ rank 0 sends and saves data

    data_dict={}

    if rank==0:

        if destfile.endswith('.mat'):
            click.echo('Will save to MAT-file: '+destfile)
        elif destfile.endswith('.z'):
            click.echo('Will save compressed pickle: '+destfile)
        elif destfile.endswith('.pkl') or destfile.endswith('.p'):
            click.echo('Will save pickle: '+destfile)
        elif destfile.endswith('.h5'):
            raise NotImplementedError
        else:
            raise Exception('Output file format not understood')

        with open(paramfile, 'r') as f:
            config = yaml.load(f.read())

        [scan_dicts,scan_parameter,scan_values,other_parameters,scan_settings]=read_cli_config(paramfile)

        if type(scan_parameter) is list:
            for param,values in zip(scan_parameter,scan_values):
                click.echo('Will scan parameter '+param)
                click.echo('Will scan value '+str(values))
        else:
            click.echo('Will scan parameter '+scan_parameter)
            click.echo('Will scan value '+str(scan_values[0]))

        # get them pc's

        data_dict=load_pcs(filename=inputfile,varname=varname,pcs=10)

        # use a list of dicts, with everything formatted ready to go

        click.echo('Whitening the training data')

        data_dict=whiten_all(data_dict)

        worker_dicts=[]

        all_keys=data_dict.keys()
        nsplits=len(data_dict)
        nparameters=len(scan_dicts)

        for cv_idx, test_key in enumerate(all_keys):
            train_keys = [key for key in all_keys if key not in test_key]
            for scan_idx, use_dict in enumerate(scan_dicts):
                for restart_idx in xrange(restarts):
                    worker_dicts.append({'scan_dict': use_dict,
                                        'index': (restart_idx,cv_idx,scan_idx),
                                        'train_keys': train_keys,
                                        'test_key': test_key,
                                        'other_parameters':other_parameters})

        # each worker gets a dictionary, the tuple index points to where the data will end up

        heldout_ll=np.zeros((restarts,nsplits,nparameters),np.float64)

        # skip the labels for now, don't need for cv

    data_dict = comm.bcast(data_dict,root=0)

    if rank==0:

        click.echo('Starting training...')
        task_index = 0
        num_workers = size - 1
        closed_workers = 0
        pbar = tqdm(total=len(worker_dicts),smoothing=0)

        while closed_workers < num_workers:

            # take 'em as they're available, farm out the job

            data = comm.recv(source=MPI.ANY_SOURCE, tag=MPI.ANY_TAG, status=status)
            source = status.Get_source()
            tag=status.Get_tag()

            if tag == tags.READY:

                if task_index < len(worker_dicts):
                    comm.send(worker_dicts[task_index],dest=source, tag=tags.START)
                    task_index+=1
                else:
                    comm.send(None, dest=source, tag=tags.EXIT)

            elif tag == tags.DONE:

                # sort out the data brutha

                worker_idx=data['index']

                # no need to export labels, requires way too much memory anyway

                heldout_ll[worker_idx]=data['heldout_ll']

                pbar.update(1)


            elif tag == tags.EXIT:

                closed_workers += 1

        pbar.close()

    else:

        while True:

            comm.send(None, dest=0, tag=tags.READY)
            worker_dict = comm.recv(source=0, tag=MPI.ANY_TAG, status=status)
            tag = status.Get_tag()

            if tag == tags.START:

                # if we get marching orders, fire up the task

                tmp_parameters=merge_dicts(worker_dict['other_parameters'],worker_dict['scan_dict'])
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

    comm.Barrier()

    if rank==0:

        click.echo('Saving results to '+destfile)
        export_dict=dict({'heldout_ll':heldout_ll,'scan_dicts':scan_dicts,'other_parameters':other_parameters},**scan_settings)
        save_dict(filename=destfile,obj_to_save=export_dict)

#TODO:  command line model training (w/ restarts, etc)


@cli.command()
@click.argument("paramfile", type=click.Path(exists=True))
@click.argument("inputfile", type=click.Path(exists=True))
@click.argument("destfile", type=click.Path(dir_okay=True,writable=True))
@click.option("--num-iter", "-n", type=int, default=100)
@click.option("--restarts", "-r", type=int, default=1)
@click.option('--varname', type=str, default='features')
@click.option('--save-every', "-s", type=int, default=1)
@click.option('--model-progress',"-p",is_flag=True)
def learn_model(paramfile, inputfile, destfile, num_iter, restarts, varname, save_every, model_progress):

    warnings.filterwarnings("ignore", category=np.VisibleDeprecationWarning)
    tags = enum('READY','DONE','EXIT','START')

    comm = MPI.COMM_WORLD
    size  = comm.Get_size()
    rank = comm.Get_rank()
    status = MPI.Status()

    # node w/ rank 0 sends and saves data

    data_dict={}

    if rank==0:

        if destfile.endswith('.mat'):
            click.echo('Will save to MAT-file: '+destfile)
        elif destfile.endswith('.z'):
            click.echo('Will save compressed pickle: '+destfile)
        elif destfile.endswith('.pkl') or destfile.endswith('.p'):
            click.echo('Will save pickle: '+destfile)
        elif destfile.endswith('.h5'):
            raise NotImplementedError
        else:
            raise Exception('Output file format not understood')

        [scan_dicts,scan_parameter,scan_values,other_parameters,scan_settings]=read_cli_config(paramfile)

        data_dict=load_pcs(filename=inputfile, varname=varname, pcs=10)

        click.echo('Whitening the training data')

        data_dict=whiten_all(data_dict)

        worker_dicts=[]
        nframes=[tmp.shape[0] for key,tmp in data_dict.iteritems()]

        # pre-initialize everything

        labels = np.empty((restarts, len(scan_dicts), len(data_dict)), dtype=object)
        loglikes = np.zeros((restarts, len(scan_dicts), np.floor(num_iter/save_every)), dtype=np.float64)
        models = np.empty((restarts,),dtype=object)

        for restart_idx in xrange(restarts):
            worker_dicts.append({'index': restart_idx,'parameters': other_parameters})

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
                    comm.send(worker_dicts[task_index], dest=source, tag=tags.START)
                    task_index += 1
                else:
                    comm.send(None, dest=source, tag=tags.EXIT)

            elif tag == tags.DONE:

                # sort out the data brutha

                worker_idx = data['index']

                if type(data['labels']) is float and np.isnan(data['labels']):
                    labels[worker_idx[0]][:]=np.nan
                else:
                    for label_itr,label in enumerate(data['labels']):
                        labels[(worker_idx[0],label_itr)]=label

                loglikes[worker_idx[0],:] = data['loglikes']
                models[worker_idx[0]] = data['model']

                pbar.update(1)

            elif tag == tags.EXIT:

                closed_workers += 1


        pbar.close()

        # if restarts>1:
        #
        #     # yeah it's ugly, sue me
        #
        #     labels_to_save=np.empty(labels.shape[1:],dtype=object)
        #     loglikes_to_save=np.zeros(loglikes.shape[1:],dtype=np.float64)
        #     max_loglikes=np.max(loglikes,axis=2)
        #     best_models=np.argmax(max_loglikes,axis=0)
        #
        #     for i in xrange(len(labels_to_save)):
        #         loglikes_to_save[i,:]=loglikes[best_models[i],i,:]
        #         for j in xrange(len(labels_to_save[0])):
        #             labels_to_save[i][j]=labels[best_models[i]][i][j]
        # else:
        #
        #     labels_to_save=np.squeeze(labels)
        #     loglikes_to_save=np.squeeze(loglikes)

        click.echo('Saving results to '+destfile)
        export_dict=dict({'loglikes': loglikes, 'labels': labels,
                          'models': models})
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

                arhmm=ARHMM(data_dict=data_dict, **worker_dict['parameters'])
                [model,loglikes,labels]=train_model(model=arhmm,
                                                    num_iter=num_iter,
                                                    save_every=save_every,
                                                    cli=True,
                                                    position=rank,
                                                    disable=not model_progress)
                comm.send({'index':worker_dict['index'],
                    'loglikes':loglikes,'labels':labels,'model':copy_model(model)}, dest=0, tag=tags.DONE)

            elif tag == tags.EXIT:

                break

        comm.send(None, dest=0, tag=tags.EXIT)

if __name__ == '__main__':
    cli()
