import click
import os
import sys
import random
import warnings
import numpy as np
from pathlib import Path
from copy import deepcopy
from cytoolz import pluck
from moseq2_model.train.util import train_model, whiten_all, whiten_each, run_e_step
from moseq2_model.util import (save_dict, load_pcs, get_parameters_from_model, copy_model,
                               load_arhmm_checkpoint, flush_print)
from ruamel.yaml import YAML
import ruamel.yaml as yaml
from collections import OrderedDict
from moseq2_model.train.models import ARHMM
from moseq2_model.train.util import train_model, whiten_all, whiten_each, run_e_step
from moseq2_model.util import (save_dict, load_pcs, get_parameters_from_model, copy_model,
                               load_arhmm_checkpoint, flush_print)
import matplotlib.pyplot as plt

def count_frames_command(input_file, var_name):

    data_dict, data_metadata = load_pcs(filename=input_file, var_name=var_name,
                                        npcs=10, load_groups=False)
    total_frames = 0
    for v in data_dict.values():
        idx = (~np.isnan(v)).all(axis=1)
        total_frames += np.sum(idx)

    print('Total frames: {}'.format(total_frames))
    return True


def learn_model_command(input_file, dest_file, config_file, index, hold_out, nfolds, num_iter,
                max_states, npcs, kappa,
                separate_trans, robust, checkpoint_freq, percent_split=20, output_directory=None):

    alpha = 5.7
    gamma = 1e3

    with open(config_file, 'r') as f:
        config_data = yaml.safe_load(f)

    # TODO: graceful handling of extra parameters:  orchestraconfig_data['ting'] this fails catastrophically if we pass
    # an extra option, just flag it to the user and ignore
    if output_directory is None:
        dest_file = os.path.realpath(dest_file)
    else:
        dest_file = os.path.join(output_directory, dest_file)


    # if not os.path.dirname(dest_file):
    #     dest_file = os.path.join('./', dest_file)

    if not os.access(os.path.dirname(dest_file), os.W_OK):
        raise IOError('Output directory is not writable.')

    if config_data['save_every'] < 0:
        click.echo("Will only save the last iteration of the model")
        save_every = num_iter + 1

    if checkpoint_freq < 0:
        checkpoint_freq = num_iter + 1

    click.echo("Entering modeling training")

    run_parameters = deepcopy(config_data)
    data_dict, data_metadata = load_pcs(filename=input_file,
                                        var_name=config_data['var_name'],
                                        npcs=npcs,
                                        load_groups=separate_trans)

    # if we have an index file, strip out the groups, match to the scores uuids
    if os.path.exists(index):
        yml = YAML(typ="rt")
        with open(index, "r") as f:
            yml_metadata = yml.load(f)["files"]
            yml_groups, yml_uuids = zip(*pluck(['group', 'uuid'], yml_metadata))

        data_metadata["groups"] = []
        for uuid in data_metadata["uuids"]:
            if uuid in yml_uuids:
                data_metadata["groups"].append(yml_groups[yml_uuids.index(uuid)])
            else:
                data_metadata["groups"].append(config_data['default_group'])

    all_keys = list(data_dict.keys())
    nkeys = len(all_keys)

    if kappa is None:
        total_frames = 0
        for v in data_dict.values():
            idx = (~np.isnan(v)).all(axis=1)
            total_frames += np.sum(idx)
        flush_print(f'Setting kappa to the number of frames: {total_frames}')
        kappa = total_frames

    if hold_out and nkeys >= nfolds:
        click.echo(f"Will hold out 1 fold of {nfolds}")

        if config_data['hold_out_seed'] >= 0:
            click.echo(f"Settings random seed to {config_data['hold_out_seed']}")
            splits = np.array_split(random.Random(config_data['hold_out_seed']).sample(list(range(nkeys)), nkeys), nfolds)
        else:
            warnings.warn("Random seed not set, will choose a different test set each time this is run...")
            splits = np.array_split(random.sample(list(range(nkeys)), nkeys), nfolds)

        hold_out_list = [all_keys[k] for k in splits[0].astype('int').tolist()]
        train_list = [k for k in all_keys if k not in hold_out_list]
        click.echo("Holding out "+str(hold_out_list))
        click.echo("Training on "+str(train_list))
    else:
        hold_out = False
        hold_out_list = None
        train_list = all_keys
        test_data = None

    if config_data['ncpus'] > len(train_list):
        ncpus = len(train_list)
        config_data['ncpus'] = ncpus
        warnings.warn(f'Setting ncpus to {nkeys}, ncpus must be <= nkeys in dataset, {len(train_list)}')

    # use a list of dicts, with everything formatted ready to go
    model_parameters = {
        'gamma': gamma,
        'alpha': alpha,
        'kappa': kappa,
        'nlags': config_data['nlags'],
        'robust': robust,
        'max_states': max_states,
        'separate_trans': separate_trans
    }

    if separate_trans:
        model_parameters['groups'] = data_metadata['groups']
    else:
        model_parameters['groups'] = None

    if config_data['whiten'][0].lower() == 'a':
        click.echo('Whitening the training data using the whiten_all function')
        data_dict = whiten_all(data_dict)
    elif config_data['whiten'][0].lower() == 'e':
        click.echo('Whitening the training data using the whiten_each function')
        data_dict = whiten_each(data_dict)
    else:
        click.echo('Not whitening the data')

    if config_data['noise_level'] > 0:
        click.echo('Using {} STD AWGN'.format(config_data['noise_level']))
        for k, v in data_dict.items():
            data_dict[k] = v + np.random.randn(*v.shape) * config_data['noise_level']

    if hold_out:
        train_data = OrderedDict((i, data_dict[i]) for i in all_keys if i in train_list)
        test_data = OrderedDict((i, data_dict[i]) for i in all_keys if i in hold_out_list)
        train_list = list(train_data.keys())
        hold_out_list = list(test_data.keys())
    else:
        train_data = data_dict
        train_list = list(data_dict.keys())

        training_data = OrderedDict()
        validation_data = OrderedDict()

        for k, v in train_data.items():
            # train values
            # print(v[int(v.shape[0]/10):], len(v[int(v.shape[0]/10):]))
            training_data[k] = np.asarray(v[int(v.shape[0] * (percent_split / 100)):])

            # validation values
            validation_data[k] = np.asarray(v[-int(v.shape[0] * (percent_split / 100)):])

    loglikes = []
    labels = []
    save_parameters = []

    checkpoint_file = dest_file+'-checkpoint.arhmm'
    # back-up file
    checkpoint_file_backup = dest_file + '-checkpoint_backup.arhmm'
    resample_save_file = dest_file + '-resamples.p'

    if os.path.exists(checkpoint_file) or os.path.exists(checkpoint_file_backup):
        flush_print('Loading Checkpoint')
        try:
            checkpoint = load_arhmm_checkpoint(checkpoint_file, train_data)
        except (FileNotFoundError, ValueError):
            flush_print('Loading original checkpoint failed, checking backup')
            if os.path.exists(checkpoint_file_backup):
                checkpoint_file = checkpoint_file_backup
            checkpoint = load_arhmm_checkpoint(checkpoint_file, train_data)
        arhmm = checkpoint.pop('model')
        itr = checkpoint.pop('iter')
        flush_print('On iteration', itr)
    else:
        arhmm = ARHMM(data_dict=train_data, **model_parameters)
        itr = 0

    progressbar_kwargs = {
        'total': num_iter,
        'cli': True,
        'file': sys.stdout,
        'leave': False,
        'disable': not config_data['progressbar'],
        'initial': itr
    }

    arhmm, loglikes_sample, labels_sample, iter_lls, iter_holls = train_model(
        model=arhmm,
        save_every=save_every,
        num_iter=num_iter,
        ncpus=ncpus,
        checkpoint_freq=checkpoint_freq,
        save_file=resample_save_file,
        checkpoint_file=checkpoint_file,
        start=itr,
        progress_kwargs=progressbar_kwargs,
        num_sessions=len(train_data.values()),
        val_data=test_data,
        separate_trans=separate_trans,
    )

    print("Iteration Training Syllable Likelihoods")
    print(iter_lls, len(iter_lls))

    if test_data is not None:
        print("Iteration Validation Syllable Likelihoods\n", iter_holls, len(iter_holls))

    ## Graph training summary
    iterations = [i for i in range(len(iter_lls))]
    plt.plot(iterations, iter_lls, color='b')
    plt.plot(iterations, iter_holls, color='r')
    plt.legend(['train ll', 'validation ll'])
    plt.xlabel('Iterations')
    plt.ylabel('Log-Likelihood')

    if hold_out:
        plt.title('ARHMM Training Summary With '+str(nfolds), ' Folds')
        plt.savefig('train_heldout_summary.png')
    else:
        plt.title('ARHMM Training Summary With '+str(percent_split)+'% Train-Val Split')
        plt.savefig('train_validation_summary.png')

    click.echo('Computing likelihoods on each training dataset...')
    if separate_trans:
        train_ll = [arhmm.log_likelihood(v, group_id=g) for g, v in zip(data_metadata['groups'], train_data.values())]
    else:
        train_ll = [arhmm.log_likelihood(v) for v in train_data.values()]
    heldout_ll = []

    if hold_out and separate_trans:
        click.echo('Computing held out likelihoods with separate transition matrix...')
        heldout_ll += [arhmm.log_likelihood(v, group_id=g) for g, v in
                       zip(data_metadata['groups'], test_data.values())]
    elif hold_out:
        click.echo('Computing held out likelihoods...')
        heldout_ll += [arhmm.log_likelihood(v) for v in test_data.values()]

    loglikes.append(loglikes_sample)
    labels.append(labels_sample)
    save_parameters.append(get_parameters_from_model(arhmm))

    # if we save the model, don't use copy_model which strips out the data and potentially
    # leaves useless certain functions we'll want to use in the future (e.g. cross-likes)
    if config_data['e_step']:
        flush_print('Running E step...')
        expected_states = run_e_step(arhmm)

    # TODO:  just compute cross-likes at the end and potentially dump the model (what else
    # would we want the model for hm?), though hard drive space is cheap, recomputing models is not...

    export_dict = {
        'loglikes': loglikes,
        'labels': labels,
        'keys': all_keys,
        'heldout_ll': heldout_ll,
        'model_parameters': save_parameters,
        'run_parameters': run_parameters,
        'metadata': data_metadata,
        'model': copy_model(arhmm) if config_data['save_model'] else None,
        'hold_out_list': hold_out_list,
        'train_list': train_list,
        'train_ll': train_ll
    }

    if config_data['e_step']:
        export_dict['expected_states'] = expected_states

    save_dict(filename=str(dest_file), obj_to_save=export_dict)

    return 'ARHMM Trained Successfully.'