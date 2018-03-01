from __future__ import division
from __future__ import print_function
import click
import os
import datetime
import subprocess
import ast
import joblib
import sys
from moseq2_model.train.models import ARHMM
import ruamel.yaml as yaml
import numpy as np
import uuid
from collections import OrderedDict
from moseq2_model.train.util import train_model, whiten_all, whiten_each
from moseq2_model.util import save_dict, load_pcs, read_cli_config,\
 get_parameters_from_model, merge_dicts, progressbar, list_rank, copy_model
from moseq2_model.kube.util import make_kube_yaml, kube_cluster_check, kube_check_mount


@click.group()
def cli():
    pass


# this will take some parameter scan specification and create a yaml file we can pipe into kubectl
@cli.command(name="parameter-scan")
@click.argument("param_file", type=click.Path(exists=True))
@click.option("--cross-validate", "-c", is_flag=True)
@click.option("--hold-out", is_flag=True)
@click.option("--nfolds", type=int, default=5)
@click.option("--num-iter", "-n", type=int, default=100)
@click.option("--restarts", "-r", type=int, default=1)
@click.option("--var_name", type=str, default='features')
@click.option("--save-every", "-s", type=int, default=1)
@click.option("--save-model", "-m", is_flag=True)
@click.option("--model-progress", "-p", is_flag=True)
@click.option("--npcs", type=int, default=10)
@click.option("--separate_trans", type=bool, default=False)
@click.option("--whiten", "-w", type=str, default='all')
@click.option("--image", "-i", type=str, envvar='MOSEQ2_GKE_MODEL_IMAGE', default='model-image')
@click.option("--job-name", type=str, default="kubejob")
@click.option("--output-dir", type=str, default="")
@click.option("--ext", "-e", type=str, default=".p.z")
@click.option("--mount-point", type=str, envvar='MOSEQ2_GKE_MOUNT_POINT', default='/mnt/user_gcs_bucket')
@click.option("--bucket", "-b", type=str, envvar='MOSEQ2_GKE_MODEL_BUCKET', default='bucket')
@click.option("--restart-policy", type=str, default="OnFailure")
@click.option("--ncpus", type=int, envvar='MOSEQ2_GKE_MODEL_NCPUS', default=4)
@click.option("--nmem", type=int, envvar='MOSEQ2_GKE_MODEL_NMEM', default=3000)
@click.option("--input-file", type=str, default="use_data.mat")
@click.option("--check-cluster", type=str, envvar='MOSEQ2_GKE_CLUSTER_NAME')
@click.option("--log-path", type=click.Path(exists=True), envvar='MOSEQ2_GKE_LOG_PATH')
@click.option("--kind", type=str, envvar='MOSEQ2_GKE_MODEL_KIND', default='Job')
@click.option("--preflight", is_flag=True)
@click.option("--copy-log", "-l", is_flag=True)
@click.option("--skip-checks", is_flag=True)
@click.option("--start-num", type=int, default=0)
def parameter_scan(param_file, cross_validate, hold_out, nfolds, num_iter, restarts, var_name, save_every,
                   save_model, model_progress, npcs, separate_trans, whiten, image, job_name,
                   output_dir, ext, mount_point, bucket, restart_policy, ncpus, nmem, input_file,
                   check_cluster, log_path, kind, preflight, copy_log, skip_checks, start_num):

    # TODO: allow for "inner" and "outer" restarts (one internal to learn model the other external)

    # use pyyaml to build up a list of worker dictionaries, make a giant yaml
    # file that we can then farm out to Kubernetes cluster using kubectl

    cfg = read_cli_config(param_file, suppress_output=True)

    suffix = '_{:%Y-%m-%d_%H-%M-%S}'.format(datetime.datetime.now())

    gcs_options = '-o allow_other --file-mode=777 --dir-mode=777'

    job_spec = locals()
    job_spec = merge_dicts(job_spec, cfg)

    if len(check_cluster) > 0 and not skip_checks:
        cluster_info = kube_cluster_check(check_cluster, ncpus=ncpus, image=image, preflight=preflight)

    if preflight and not skip_checks:
        pass_flag, _ = kube_check_mount(**job_spec)
        if preflight:
            return None

    yaml_out, output_dicts, output_dir, bucket_dir = make_kube_yaml(**job_spec)

    # send the yaml to stdout

    click.echo(yaml_out)
    job_spec.pop('cfg', None)
    job_spec.pop('worker_dicts', None)
    job_spec['worker_dicts'] = output_dicts
    # represent_dict_order = lambda self, data:  self.represent_mapping('tag:yaml.org,2002:map', data.items())
    # yaml.RoundTripDumper.add_representer(OrderedDict, represent_ordereddict)

    # job_spec =
    # job_spec = OrderedDict((str(key), str(value)) for key, value in sorted(job_spec.iteritems()))

    if log_path is None:
        log_path = os.getcwd()

    # copy yaml file to log directory as well

    log_store_path = os.path.join(log_path, job_name+suffix+'.yaml')
    with open(log_store_path, 'w') as f:
        yaml.dump(job_spec, f)

    # is user specifies copy this ish to the output directory as well for solid(!) bookkeeping

    if copy_log and bucket_dir:
        subprocess.check_output("gsutil cp "+log_store_path +
                                " gs://"+os.path.join(bucket_dir, 'job_manifest.yaml'), shell=True)


# this is the entry point for learning models over Kubernetes, expose all
# parameters we could/would possibly scan over
@cli.command(name="learn-model")
@click.argument("input_file", type=click.Path(exists=True))
@click.argument("dest_file", type=click.Path(dir_okay=True, writable=True))
@click.option("--hold-out", "-h", type=int, default=None, help="Index of data group to hold out (<nfolds)")
@click.option("--nfolds", type=int, default=None, help="Number of folds for split")
@click.option("--num-iter", "-n", type=int, default=100, help="Number of times to resample model")
@click.option("--restarts", "-r", type=int, default=1, help="Number of restarts for model")
@click.option("--var-name", type=str, default='features', help="Variable name in input file with PCs")
@click.option("--save-every", "-s", type=int, default=-1,
              help="Increment to save labels and model object (-1 for just last)")
@click.option("--save-model", is_flag=True, help="Save model object")
@click.option("--max-states", "-m", type=int, default=100, help="Maximum number of states")
@click.option("--model-progress", "-p", is_flag=True, help="Show model progress")
@click.option("--npcs", type=int, default=10, help="Number of PCs to use")
@click.option("--whiten", "-w", type=str, default='all', help="Whiten (e)each (a)ll or (n)o whitening")
@click.option("--kappa", "-k", type=float, default=1e8, help="Kappa")
@click.option("--gamma", "-g", type=float, default=1e3, help="Gamma")
@click.option("--nu", type=float, default=4, help="Nu (only applicable if robust set to true)")
@click.option("--nlags", type=int, default=3, help="Number of lags to use")
@click.option("--separate-trans", is_flag=True, help="Use separate transition matrix per group")
@click.option("--robust", is_flag=True, help="Use tAR model")
def learn_model(input_file, dest_file, hold_out, nfolds, num_iter, restarts, var_name,
                save_every, save_model, max_states, model_progress, npcs, whiten,
                kappa, gamma, nu, nlags, separate_trans, robust):

    # TODO: graceful handling of extra parameters:  orchestrating this fails catastrophically if we pass
    # an extra option, just flag it to the user and ignore

    if save_every < 0:
        click.echo("Will only save the last iteration of the model")
        save_every = num_iter+1

    click.echo("Entering modeling training")

    run_parameters = locals()
    data_dict, data_metadata = load_pcs(filename=input_file, var_name=var_name,
                                        npcs=npcs, load_groups=separate_trans)
    all_keys = data_dict.keys()
    nkeys = len(all_keys)

    if hold_out and hold_out >= 0 and nfolds >= hold_out and nkeys >= nfolds:
        click.echo("Will hold out split "+str(hold_out)+" of "+str(nfolds))
        splits = np.array_split(range(nkeys), nfolds)
        hold_out_list = [all_keys[i] for i in splits[hold_out].astype('int').tolist()]
        train_list = [all_keys[i] for i in all_keys if i not in hold_out_list]
        click.echo("Holding out indices "+str(hold_out_list))
        click.echo("Training on indices"+str(train_list))

    # use a list of dicts, with everything formatted ready to go

    model_parameters = {
        'gamma': gamma,
        'kappa': kappa,
        'nlags': nlags,
        'separate_trans': separate_trans,
        'robust': robust,
        'max_states': max_states,
        'nu': nu
    }

    if separate_trans:
        model_parameters['groups'] = data_metadata['groups']
    else:
        model_parameters['groups'] = None

    if whiten[0].lower() == 'a':
        click.echo('Whitening the training data using the whiten_all function')
        data_dict = whiten_all(data_dict)
    elif whiten[0].lower() == 'e':
        click.echo('Whitening the training data using the whiten_each function')
        data_dict = whiten_each(data_dict)
    else:
        click.echo('Not whitening the data')

    if hold_out >= 0 and nfolds >= hold_out:
        train_data = OrderedDict((i, data_dict[i]) for i in all_keys if i in train_list)
        test_data = OrderedDict((i, data_dict[i]) for i in all_keys if i in hold_out_list)
    else:
        train_data = data_dict
        test_data = None

    loglikes = []
    labels = []
    heldout_ll = []
    save_parameters = []

    for i in range(restarts):
        arhmm = ARHMM(data_dict=train_data, **model_parameters)
        [arhmm, loglikes_sample, labels_sample] = \
            train_model(model=arhmm,
                        save_every=save_every,
                        num_iter=num_iter,
                        cli=True,
                        leave=False,
                        disable=not model_progress,
                        total=num_iter*restarts,
                        initial=i*num_iter,
                        file=sys.stdout)

        if test_data:
            click.echo("Computing held out likelihoods...")
            [heldout_ll.append(arhmm.log_likelihood(v)) for k, v in test_data.iteritems()]

        loglikes.append(loglikes_sample)
        labels.append(labels_sample)
        save_parameters.append(get_parameters_from_model(arhmm))

    # if we save the model, don't use copy_model which strips out the data and potentially
    # leaves useless certain functions we'll want to use in the future (e.g. cross-likes)

    if save_model:
        save_model = copy_model(arhmm)
    else:
        save_model = None

    # TODO:  just compute cross-likes at the end and potentially dump the model (what else
    # would we want the model for hm?), though hard drive space is cheap, recomputing models is not...

    # TODO: decision time, we could just save the model and strip out the parameters later,
    # would be much more lightweight, right now we're being too redundant

    export_dict = {
        'loglikes': loglikes,
        'labels': labels,
        'heldout_ll': heldout_ll,
        'model_parameters': save_parameters,
        'run_parameters': run_parameters,
        'metadata': data_metadata,
        'model': save_model
        }

    save_dict(filename=dest_file, obj_to_save=export_dict)


@cli.command("export-results")
@click.option("--input-dir", "-i", type=click.Path(exists=True), default=os.getcwd())
@click.option("--job-manifest", "-j", type=click.Path(exists=True, readable=True),
              default=os.path.join(os.getcwd(), 'job_manifest.yaml'))
@click.option("--dest-file", "-d", type=click.Path(dir_okay=True, writable=True),
              default=os.path.join(os.getcwd(), 'export_results.mat'))
def export_results(input_dir, job_manifest, dest_file):

    # TODO: smart detection of restarts and cross-validation (use worker_dicts or job manifest)

    with open(job_manifest, 'r') as f:
        manifest = yaml.load(f.read(), Loader=yaml.Loader)

    if type(manifest['worker_dicts']) is list:
        parse_dicts = manifest['worker_dicts']
    elif type(manifest['worker_dicts']) is str:
        parse_dicts = ast.literal_eval(manifest['worker_dicts'])

    # if 'hold-out' in parse_dicts[0].keys():
    #     for i in xrange(len(parse_dicts)):
    #         parse_dicts[i]['hold_out'] = parse_dicts[i].pop('hold-out')

    nfiles = len(parse_dicts)

    for i, use_dict in enumerate(parse_dicts):
        test_load = joblib.load(os.path.join(input_dir, os.path.basename(use_dict['filename'])))
        if (list_rank(test_load['labels']) == 1 and
                len(test_load['labels']) == 1 and np.all(np.isnan(test_load['labels'][0]))):
            continue
        else:
            rank = list_rank(test_load['labels'])
            break

    click.echo(str(rank))

    if rank == 2:
        restart_list = True
        nrestarts = len(test_load['labels'])
        nsets = len(test_load['labels'][0])
        nholdouts = len(test_load['heldout_ll'])
    elif rank < 2:
        restart_list = False
        nsets = len(test_load['labels'])
        nholdouts = len(test_load['heldout_ll'])
        nrestarts = 1
    else:
        raise ValueError("Cannot interpret labels")

    if 'metadata' in test_load.keys():
        metadata = test_load['metadata']
        for key, value in metadata.iteritems():
            print (key)
            if value is None:
                metadata[key] = 'Null'
            if key == 'uuids' or key == 'groups' and value is list and all(isinstance(value, str)):
                metadata[key] = [n.encode("utf8") for n in value]
    else:
        metadata = {}

    save_array = np.empty((nfiles, nsets, nrestarts), dtype=np.object)

    all_parameters = {
        'kappa': np.empty((nfiles, nrestarts), dtype=np.float32),
        'gamma': np.empty((nfiles, nrestarts), dtype=np.float32),
        'nu': np.empty((nfiles, nrestarts), dtype=np.float32),
        'num_states': np.empty((nfiles, nrestarts), dtype=np.uint16),
        'nlags': np.empty((nfiles, nrestarts), dtype=np.uint8),
        'npcs': np.empty((nfiles, nrestarts), dtype=np.uint8),
        'ar_mat': np.empty((nfiles, nrestarts), dtype=np.object),
        'sig': np.empty((nfiles, nrestarts), dtype=np.object)
    }

    for k in all_parameters.keys():
        all_parameters[k][:] = np.nan

    heldout_ll = np.empty((nfiles, nholdouts, nrestarts), dtype=np.float32)
    loglikes = np.empty((nfiles, nrestarts), dtype=np.float32)

    heldout_ll[:] = np.nan
    loglikes[:] = np.nan
    save_array[:] = np.nan

    # farm this out with joblib parallel
    # parse_dicts = parse_dicts[:2]

    for i, use_dict in enumerate(progressbar(parse_dicts, cli=True)):

        # scan_dicts[i] = parse_dicts[i]

        try:
            use_data = joblib.load(os.path.join(input_dir, os.path.basename(use_dict['filename'])))
        except IOError:
            continue
        except KeyboardInterrupt:
            click.echo("Bailing")
            sys.exit()

        if restart_list:

            for j in range(nrestarts):
                tmp = use_data['model_parameters'][j]
                tmp['npcs'] = tmp['sig'][0].shape[0]

                for k, v in tmp.iteritems():
                    if np.all(v is None):
                        tmp[k] = np.nan

                    if k == 'sig' or k == 'ar_mat':
                        nsigs = len(v)
                        r, c = v[0].shape
                        newsig = np.empty((nsigs, r, c), dtype=np.float32)
                        for l, sig in enumerate(v):
                            newsig[l, ...] = sig
                        tmp[k] = newsig

                for k in all_parameters.keys():
                    all_parameters[k][i, j] = tmp[k]

                for k in range(nsets):
                    save_array[i, k, j] = np.array(use_data['labels'][j][k][-1], dtype=np.int16)

                for k in range(nholdouts):
                    heldout_ll[i][k][j] = use_data['heldout_ll'][k]

                loglikes[i][j] = use_data['loglikes'][j][-1]
        else:

            tmp = use_data['model_parameters']
            for k, v in tmp.iteritems():
                if np.all(v is None):
                    tmp[k] = np.nan
            all_parameters[i][0] = tmp

            for j in xrange(nsets):
                save_array[i][j][0] = np.array(use_data['labels'][j][-1], dtype=np.int16)

            for j in range(nholdouts):
                heldout_ll[i][j] = use_data['heldout_ll'][j]

            loglikes[i][0] = use_data['loglikes'][-1]

    # export labels, parameter, bookkeeping stuff

    metadata['parameters'] = all_parameters
    metadata['export_uuid'] = str(uuid.uuid4())
    # metadata['scan_dicts'] = scan_dicts
    metadata['loglikes'] = loglikes
    metadata['heldout_ll'] = heldout_ll

    export_dict = {
        'labels': save_array,
        'metadata': metadata
        }

    # strip out the filename and put in the uuid

    filename = os.path.basename(dest_file)
    pathname = os.path.dirname(dest_file)
    ext = os.path.splitext(filename)

    new_filename = ext[0]+'_'+metadata['export_uuid']
    dest_file = os.path.join(pathname, new_filename+ext[1])

    save_dict(filename=dest_file, obj_to_save=export_dict)


if __name__ == '__main__':
    cli()
