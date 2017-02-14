from __future__ import division
import numpy as np
import click
import cPickle as pickle
import yaml
from tqdm import tqdm
from train.models import ARHMM
from collections import OrderedDict
from train.util import merge_dicts, train_model, whiten_all
from util import load_data_from_matlab, enum
from mpi4py import MPI
import scipy.io as sio
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
def parameter_scan(paramfile, inputfile, destfile, num_iter=100, restarts=5):

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
        elif destfile.endswith('.pklz') | destfile.endswidth('.pz'):
            click.echo('Will save gzipped pickle: '+destfile)
        elif destfile.endswith('.pkl') | destfile.endswith('.p'):
            click.echo('Will save pickle: '+destfile)
        elif destfile.endswith('.h5'):
            raise NotImplementedError

        with open(paramfile, 'r') as f:
            config = yaml.load(f.read())

        scan_settings = config['scan_settings']
        scan_parameter = scan_settings['scan_parameter']
        scan_range =  scan_settings['scan_range']
        scan_scale =  scan_settings['scan_scale']

        if scan_scale=='log':
            scan_values=np.logspace(*scan_range)
        elif scan_scale=='linear':
            scan_values=np.linspace(*scan_range)

        #scan_parameter = config['scan_parameter']
        #tmp_scan = config['scan_values']
        #scan_values=[x for x in tmp_scan]

        click.echo('Will scan parameter '+scan_parameter)
        click.echo('Scan values '+str(scan_values))

        other_parameters={}

        if 'parameters' in config.keys():
            other_parameters=config['parameters']

        # get them pc's

        if inputfile.endswith('.mat'):
            data_dict=load_data_from_matlab(inputfile)
        elif inputfile.endswidth('.p'):
            data_dict=pickle.load(open("data_dict.inputfile","rb"))

        # use a list of dicts, with everything formatted ready to go

        click.echo('Whitening the training data')

        data_dict=whiten_all(data_dict)

        worker_dicts=[]

        for scan_idx,scan_value in enumerate(scan_values):
            for restart_idx in xrange(restarts):
                worker_dicts.append({'scan_parameter': scan_parameter,
                    'scan_value':scan_value,
                    'index': (restart_idx,scan_idx),
                    'other_parameters':other_parameters})

        # each worker gets a dictionary, the tuple index points to where the data will end up

        labels=np.empty((restarts,len(scan_values),len(data_dict)),dtype=object)
        loglikes=np.empty((restarts,len(scan_values)),dtype=object)

    data_dict = comm.bcast(data_dict,root=0)

    if rank==0:

        click.echo('Starting training...')

        task_index = 0
        num_workers = size - 1
        closed_workers = 0
        pbar = tqdm(total=len(worker_dicts))

        while closed_workers < num_workers:

            # take 'em as they're available, farm out the job

            data = comm.recv(source=MPI.ANY_SOURCE, tag=MPI.ANY_TAG, status=status)
            source = status.Get_source()
            tag=status.Get_tag()

            if tag == tags.READY:

                if task_index < len(worker_dicts):
                    #click.echo('Distributing task '+str(task_index))
                    comm.send(worker_dicts[task_index],dest=source, tag=tags.START)
                    task_index+=1
                else:
                    comm.send(None, dest=source, tag=tags.EXIT)

            elif tag == tags.DONE:

                # sort out the data brutha

                #click.echo('Worker '+str(source)+' finished')
                worker_idx=data['index']

                tmp_labels=data['labels']
                for label_itr,label in enumerate(tmp_labels):
                    labels[(worker_idx[0],worker_idx[1],label_itr)]=label

                loglikes[worker_idx]=data['loglikes']
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
                tmp_parameters=merge_dicts(worker_dict['other_parameters'],{worker_dict['scan_parameter']:worker_dict['scan_value']})
                arhmm=ARHMM(data_dict=data_dict, **tmp_parameters)
                [arhmm,loglikes,labels]=train_model(model=arhmm,num_iter=num_iter, num_procs=1, cli=True)
                comm.send({'index':worker_dict['index'],
                    'loglikes':loglikes,'labels':labels}, dest=0, tag=tags.DONE)

            elif tag == tags.EXIT:

                break

        comm.send(None, dest=0, tag=tags.EXIT)

    comm.Barrier()

    if rank==0:

        click.echo('Saving results to '+destfile)

        if destfile.endswith('.mat'):
            sio.savemat(destfile,mdict={'loglikes':loglikes,'labels':labels})
        elif destfile.endswith('.pklz') | destfile.endswidth('.pz'):
            # pickle it
            with gzip.open(destfile, 'w') as outfile:
                pickle.dump({'loglikes': loglikes, 'labels': labels},
                    outfile, protocol=-1)
        elif destfile.endswith('.pkl') | destfile.endswith('.p'):
            # pickle it
            with open(destfile, 'wb') as outfile:
                pickle.dump({'loglikes': loglikes, 'labels': labels},
                    outfile, protocol=-1)
        elif destfile.endswith('.h5'):
            raise NotImplementedError

@cli.command()
@click.argument("paramfile", type=click.Path(exists=True))
@click.argument("inputfile", type=click.Path(exists=True))
@click.argument("destfile", type=click.Path(dir_okay=True,writable=True))
@click.option("--num-iter", "-n", type=int, default=100)
@click.option("--restarts", "-r", type=int, default=1)
def cv_parameter_scan(paramfile, inputfile, destfile, num_iter=100, restarts=5):

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
        elif destfile.endswith('.pklz') | destfile.endswidth('.pz'):
            click.echo('Will save gzipped pickle: '+destfile)
        elif destfile.endswith('.pkl') | destfile.endswith('.p'):
            click.echo('Will save pickle: '+destfile)
        elif destfile.endswith('.h5'):
            raise NotImplementedError

        with open(paramfile, 'r') as f:
            config = yaml.load(f.read())

        scan_settings = config['scan_settings']
        scan_parameter = scan_settings['scan_parameter']
        scan_range =  scan_settings['scan_range']
        scan_scale =  scan_settings['scan_scale']

        if scan_scale=='log':
            scan_values=np.logspace(*scan_range)
        elif scan_scale=='linear':
            scan_values=np.linspace(*scan_range)

        click.echo('Will scan parameter '+scan_parameter)
        click.echo('Scan values '+str(scan_values))

        other_parameters={}

        if 'parameters' in config.keys():
            other_parameters=config['parameters']

        # get them pc's

        if inputfile.endswith('.mat'):
            data_dict=load_data_from_matlab(inputfile)
        elif inputfile.endswidth('.p'):
            data_dict=pickle.load(open("data_dict.inputfile","rb"))

        # use a list of dicts, with everything formatted ready to go

        click.echo('Whitening the training data')

        data_dict=whiten_all(data_dict)

        worker_dicts=[]

        all_keys=data_dict.keys()
        nsplits=len(data_dict)
        nparameters=len(scan_values)

        for cv_idx,test_key in enumerate(all_keys):
            train_keys=[key for key in all_keys if key not in test_key]
            for scan_idx,scan_value in enumerate(scan_values):
                for restart_idx in xrange(restarts):
                    worker_dicts.append({'scan_parameter': scan_parameter,
                        'scan_value':scan_value,
                        'index': (restart_idx,cv_idx,scan_idx),
                        'train_keys': train_keys,
                        'test_key': test_key,
                        'other_parameters':other_parameters})

        # each worker gets a dictionary, the tuple index points to where the data will end up

        heldout_ll=np.empty((restarts,nsplits,nparameters),np.float64)
        labels=np.empty((restarts,nsplits,nparameters,len(data_dict)-1),dtype=object)

    data_dict = comm.bcast(data_dict,root=0)

    if rank==0:

        click.echo('Starting training...')
        task_index = 0
        num_workers = size - 1
        closed_workers = 0
        pbar = tqdm(total=len(worker_dicts))

        while closed_workers < num_workers:

            # take 'em as they're available, farm out the job

            data = comm.recv(source=MPI.ANY_SOURCE, tag=MPI.ANY_TAG, status=status)
            source = status.Get_source()
            tag=status.Get_tag()

            if tag == tags.READY:

                if task_index < len(worker_dicts):
                    #click.echo('Distributing task '+str(task_index))
                    comm.send(worker_dicts[task_index],dest=source, tag=tags.START)
                    task_index+=1
                else:
                    comm.send(None, dest=source, tag=tags.EXIT)

            elif tag == tags.DONE:

                # sort out the data brutha

                #click.echo('Worker '+str(source)+' finished')
                worker_idx=data['index']
                tmp_labels=data['labels']
                #click.echo(worker_idx)

                for label_itr,label in enumerate(tmp_labels):
                    labels[(worker_idx[0],worker_idx[1],worker_idx[2],label_itr)]=label

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

                tmp_parameters=merge_dicts(worker_dict['other_parameters'],{worker_dict['scan_parameter']:worker_dict['scan_value']})
                # click.echo(tmp_parameters)
                # click.echo(worker_dict['train_keys'])
                # click.echo(worker_dict['test_key'])
                train_data=OrderedDict((i,data_dict[i]) for i in worker_dict['train_keys'])
                arhmm=ARHMM(data_dict=train_data, **tmp_parameters)
                [arhmm,loglikes,labels]=train_model(model=arhmm,num_iter=num_iter, num_procs=1, cli=True)
                heldout_ll = arhmm.log_likelihood(data_dict[worker_dict['test_key']])
                comm.send({'index':worker_dict['index'],
                    'heldout_ll':heldout_ll,'labels':labels}, dest=0, tag=tags.DONE)

            elif tag == tags.EXIT:

                break

        comm.send(None, dest=0, tag=tags.EXIT)

    comm.Barrier()

    if rank==0:

        click.echo('Saving results to '+destfile)

        if destfile.endswith('.mat'):
            sio.savemat(destfile,mdict={'heldout_ll':heldout_ll ,'labels':labels})
        elif destfile.endswith('.pklz') | destfile.endswidth('.pz'):
            # pickle it
            with gzip.open(destfile, 'w') as outfile:
                pickle.dump({'heldout_ll': heldout_ll, 'labels': labels},
                    outfile, protocol=-1)
        elif destfile.endswith('.pkl') | destfile.endswith('.p'):
            # pickle it
            with open(destfile, 'wb') as outfile:
                pickle.dump({'heldout_ll': heldout_ll, 'labels': labels},
                    outfile, protocol=-1)
        elif destfile.endswith('.h5'):
            raise NotImplementedError



if __name__ == '__main__':
    cli()
