from __future__ import division
import click, os, datetime, subprocess, ast, joblib, gzip, sys
from tqdm import tqdm
from train.models import ARHMM
import ruamel.yaml as yaml
import numpy as np
from collections import OrderedDict
from train.util import train_model, whiten_all
from util import enum, save_dict, load_pcs, read_cli_config, copy_model,\
 get_parameters_from_model, represent_ordereddict, merge_dicts, progressbar, list_rank
from kube.util import make_kube_yaml, kube_cluster_check, kube_check_mount
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
def mpi_parameter_scan(param_file, input_file, dest_file, cross_validate,
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


@cli.command()
@click.argument("cluster_name", type=str, envvar='KINECT_GKE_CLUSTER_NAME')
def kube_print_cluster_info(cluster_name):
    # get some sweet yaml describing our cluster

    cluster_info=kube_info(cluster_name)

    click.echo(cluster_info)
    click.echo("cluster name="+cluster_info['cluster_name'])
    click.echo("ncpus="+str(cluster_info['ncpus']))
    click.echo(cluster_info['images'])

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
@click.option("--image","-i",type=str, envvar='KINECT_GKE_MODEL_IMAGE', default='model-image')
@click.option("--job-name", type=str, default="kubejob")
@click.option("--output-dir", type=str, default="")
@click.option("--ext","-e",type=str, default=".p.z")
@click.option("--mount-point", type=str, envvar='KINECT_GKE_MOUNT_POINT', default='/mnt/user_gcs_bucket')
@click.option("--bucket","-b",type=str, envvar='KINECT_GKE_MODEL_BUCKET', default='bucket')
@click.option("--restart-policy", type=str, default="OnFailure")
@click.option("--ncpus", type=int, envvar='KINECT_GKE_MODEL_NCPUS', default=4)
@click.option("--nmem", type=int, envvar='KINECT_GKE_MODEL_NMEM', default=3000)
@click.option("--input-file", type=str, default="use_data.mat")
@click.option("--check-cluster", type=str, envvar='KINECT_GKE_CLUSTER_NAME')
@click.option("--log-path", type=click.Path(exists=True), envvar='KINECT_GKE_LOG_PATH')
@click.option("--ssh-key", type=str, envvar='KINECT_GKE_SSH_KEY', default=None)
@click.option("--ssh-user", type=str, envvar='KINECT_GKE_SSH_USER', default=None)
@click.option("--ssh-remote-server", type=str, envvar='KINECT_GKE_SSH_REMOTE_SERVER', default=None)
@click.option("--ssh-remote-dir", type=str, envvar='KINECT_GKE_SSH_REMOTE_DIR', default=None)
@click.option("--ssh-mount-point",type=str, envvar='KINECT_GKE_SSH_MOUNT_POINT', default=None)
@click.option("--kind",type=str, envvar='KINECT_GKE_MODEL_KIND', default='Job')
@click.option("--preflight",is_flag=True)
@click.option("--copy-log",is_flag=True)
@click.option("--skip-checks", is_flag=True)
def kube_parameter_scan(param_file, cross_validate,
    num_iter, restarts, var_name, save_every, save_model, model_progress,npcs,whiten,
    image, job_name, output_dir, ext, mount_point, bucket, restart_policy,
    ncpus, nmem, input_file, check_cluster, log_path, ssh_key, ssh_user, ssh_remote_server,
    ssh_remote_dir, ssh_mount_point, kind, preflight, copy_log, skip_checks):

    # TODO: automatic copy of the input file to the appropriate location before the modeling
    # TODO: allow for "inner" and "outer" restarts (one internal to learn model the other external)

    # use pyyaml to build up a list of worker dictionaries, make a giant yaml
    # file that we can then farm out to Kubernetes cluster using kubectl

    cfg=read_cli_config(param_file,suppress_output=True)

    suffix='_{:%Y-%m-%d_%H-%M-%S}'.format(datetime.datetime.now())

    gcs_options='-o allow_other --file-mode=777 --dir-mode=777'

    job_spec=locals()
    job_spec=merge_dicts(job_spec,cfg)

    if len(check_cluster)>0 and not skip_checks:
        cluster_info=kube_cluster_check(check_cluster,ncpus=ncpus,image=image,preflight=preflight)

    # TODO: use stdout instead of stderr for TQDM???

    if preflight and not skip_checks or cross_validate:
        pass_flag,nfolds=kube_check_mount(**job_spec)
        job_spec['nfolds']=nfolds
        if preflight:
            return None

    yaml_out,output_dicts,output_dir,bucket_dir=make_kube_yaml(**job_spec)

    # send the yaml to stdout

    click.echo(yaml_out)
    job_spec.pop('cfg',None)
    job_spec.pop('worker_dicts',None)
    job_spec['worker_dicts']=output_dicts
    represent_dict_order = lambda self, data:  self.represent_mapping('tag:yaml.org,2002:map', data.items())
    yaml.RoundTripDumper.add_representer(OrderedDict, represent_ordereddict)

    job_spec=OrderedDict((str(key),str(value)) for key,value in sorted(job_spec.iteritems()))

    if log_path==None:
        log_path=os.getcwd()

    # copy yaml file to log directory as well

    log_store_path=os.path.join(log_path,job_name+suffix+'.yaml')
    with open(log_store_path,'w') as f:
        yaml.dump(job_spec,f,Dumper=yaml.RoundTripDumper)

    # is user specifies copy this ish to the output directory as well for solid(!) bookkeeping

    if copy_log and bucket_dir:
        subprocess.check_output("gsutil cp "+log_store_path+" gs://"+os.path.join(bucket_dir,'job_manifest.yaml'),shell=True)


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
@click.option("--save-ar",is_flag=True)
def learn_model(input_file, dest_file, hold_out, num_iter, restarts, var_name, save_every,
    save_model, model_progress, npcs, whiten, kappa, gamma, nlags,save_ar):

    # TODO: graceful handling of extra parameters:  orchestrating this fails catastrophically if we pass
    # an extra option, just flag it to the user and ignore

    click.echo("Entering modeling training")

    run_parameters=locals()
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
    else:
        train_data=data_dict

    loglikes = []
    labels = []
    heldout_ll = []
    save_parameters = []

    for i in xrange(restarts):
        arhmm=ARHMM(data_dict=train_data, **model_parameters)
        [arhmm,loglikes_sample,labels_sample]=train_model(model=arhmm,
                                        num_iter=num_iter,
                                        cli=True,
                                        leave=False,
                                        disable=not model_progress,
                                        total=num_iter*restarts,
                                        initial=i*num_iter,
                                        file=sys.stdout)

        if hold_out>=0:
            heldout_ll.append(arhmm.log_likelihood(data_dict[all_keys[hold_out]]))
        else:
            heldout_ll.append(None)

        loglikes.append(loglikes_sample)
        labels.append(labels_sample)
        save_parameters.append(get_parameters_from_model(arhmm))

    export_dict=dict({'loglikes':loglikes, 'labels':labels, 'heldout_ll':heldout_ll,
                      'model_parameters':save_parameters,'run_parameters':run_parameters})
    save_dict(filename=dest_file,obj_to_save=export_dict)


@cli.command()
@click.argument("input_dir", type=click.Path(exists=True))
@click.argument("job_manifest", type=click.Path(exists=True,readable=True))
@click.argument("dest_file","-j", type=click.Path(dir_okay=True,writable=True))
def export_results(input_dir, job_manifest, dest_file):

    # TODO: smart detection of restarts and cross-validation (use worker_dicts or job manifest)
    # TODO: include other stuff we may want, e.g. log-likes

    with open(job_manifest,'r') as f:
        manifest=yaml.load(f.read(),Loader=yaml.Loader)

    parse_dicts=ast.literal_eval(manifest['worker_dicts'])

    test_load=joblib.load(os.path.join(input_dir,os.path.basename(parse_dicts[0]['filename'])))
    nfiles=len(parse_dicts)

    rank=list_rank(test_load['labels'])
    click.echo(str(rank))
    if rank==2:
        restart_list=True
        nrestarts=len(test_load['labels'])
        nsets=len(test_load['labels'][0])
    elif rank<2:
        restart_list=False
        nsets=len(test_load['labels'])
        nrestarts=1
    else:
        raise ValueError("Cannot interpret labels")

    save_array=np.empty((nfiles,nsets,nrestarts),dtype=object)
    all_parameters=np.empty((nfiles,nrestarts,),dtype=object)
    heldout_ll=np.empty((nfiles,nrestarts),dtype=np.float64)

    for i,use_dict in enumerate(progressbar(parse_dicts, cli=True)):

        use_data=joblib.load(os.path.join(input_dir,os.path.basename(use_dict['filename'])))

        if restart_list:
            for j in xrange(nrestarts):
                all_parameters[i][j]=use_data['model_parameters'][j]

                try:
                    for k in xrange(nsets):
                        save_array[i][k][j]=np.array(use_data['labels'][j][k][-1],dtype=np.int16)
                except:
                    for k in xrange(nsets):
                        save_array[i][k][j]=np.nan

                heldout_ll[i][j]=use_data['heldout_ll'][j]

        else:

            all_parameters[i][0]=use_data['model_parameters']

            try:
                for j in xrange(nsets):
                    save_array[i][j][0]=np.array(use_data['labels'][j][-1],dtype=np.int16)
            except:
                for j in xrange(nsets):
                    save_array[i][j][0]=np.nan

            heldout_ll[i][0]=use_data['heldout_ll']

    # export labels, parameter, bookkeeping stuff

    export_dict=dict({'scan_dicts':parse_dicts,
                      'labels':save_array,
                      'parameters':all_parameters,
                      'heldout_ll':heldout_ll})

    save_dict(filename=dest_file,obj_to_save=export_dict)
