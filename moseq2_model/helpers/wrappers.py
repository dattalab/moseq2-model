'''
Wrapper functions for all functionality included in MoSeq2-Model that is accessible via CLI or GUI.

Each wrapper function executes the functionality from end-to-end given it's dependency parameters are inputted.
(See CLI Click parameters)
'''

import os
import sys
import glob
import click
from copy import deepcopy
from collections import OrderedDict
from moseq2_model.train.util import train_model, run_e_step
from os.path import join, basename, realpath, dirname, exists
from moseq2_model.util import (save_dict, load_pcs, get_parameters_from_model, copy_model,
                               get_current_model, get_loglikelihoods, get_session_groupings)
from moseq2_model.helpers.data import (process_indexfile, select_data_to_model, prepare_model_metadata, count_frames,
                                       graph_modeling_loglikelihoods, get_heldout_data_splits, get_training_data_splits)

def learn_model_wrapper(input_file, dest_file, config_data, index=None):
    '''
    Wrapper function to train ARHMM, shared between CLI and GUI.

    Parameters
    ----------
    input_file (str): path to pca scores file.
    dest_file (str): path to save model to.
    config_data (dict): dictionary containing necessary modeling parameters.
    index (str): path to index file.
    
    Returns
    -------
    None
    '''

    # TODO: graceful handling of extra parameters: orchestra this fails catastrophically if we pass
    # an extra option, just flag it to the user and ignore
    dest_file = realpath(dest_file)

    if not os.path.exists(dirname(dest_file)):
        os.makedirs(dirname(dest_file))

    if not os.access(dirname(dest_file), os.W_OK):
        raise IOError('Output directory is not writable.')

    # Handle checkpoint parameters
    checkpoint_path = join(dirname(dest_file), 'checkpoints/')
    checkpoint_freq = config_data.get('checkpoint_freq', -1)

    if checkpoint_freq < 0:
        checkpoint_freq = config_data.get('num_iter', 100) + 1
    else:
        if not exists(checkpoint_path):
            os.makedirs(checkpoint_path)

    click.echo("Entering modeling training")

    run_parameters = deepcopy(config_data)

    # Get session PC scores and session metadata dicts
    data_dict, data_metadata = load_pcs(filename=input_file,
                                        var_name=config_data.get('var_name', 'scores'),
                                        npcs=config_data['npcs'],
                                        load_groups=config_data['load_groups'])

    # Parse index file and update metadata information; namely groups
    index_data, data_metadata = process_indexfile(index, config_data, data_metadata)

    # Get all training session uuids
    all_keys = list(data_dict.keys())
    groups = list(data_metadata['groups'])

    # Get keys to include in training set
    select_groups = config_data.get('select_groups', False)
    if (index_data != None):
        all_keys, groups = select_data_to_model(index_data, select_groups)
        data_metadata['groups'] = groups
        data_metadata['uuids'] = all_keys

    # Create OrderedDict of training data
    data_dict = OrderedDict((i, data_dict[i]) for i in all_keys)
    nkeys = len(all_keys)

    # Get train/held out data split uuids
    config_data, data_dict, model_parameters, train_list, hold_out_list = \
        prepare_model_metadata(data_dict, data_metadata, config_data, nkeys, all_keys)

    # Pack data dicts corresponding to uuids in train_list and hold_out_list
    if config_data['hold_out']:
        train_data, hold_out_list, test_data, nt_frames = \
            get_heldout_data_splits(all_keys, data_dict, train_list, hold_out_list)
    else:
        # If not holding out sessions, split the data into a validation set with the percent_split option
        train_data, test_data, nt_frames = get_training_data_splits(config_data, data_dict)

    # Get all saved checkpoints
    checkpoint_file = join(checkpoint_path, basename(dest_file).replace('.p', '') + '-checkpoint.arhmm')
    all_checkpoints = [f for f in glob.glob(f'{checkpoint_path}*.arhmm') if basename(dest_file).replace('.p', '') in f]

    # Instantiate model; either anew or from previously saved checkpoint
    arhmm, itr = get_current_model(config_data.get('use_checkpoint', False), all_checkpoints, train_data, model_parameters)

    # Pack progress bar keyword arguments
    progressbar_kwargs = {
        'total': config_data['num_iter'],
        'file': sys.stdout,
        'leave': False,
        'disable': not config_data['progressbar'],
        'initial': itr
    }

    if config_data['converge']:
        config_data['num_iter'] = 1000

    # Get data groupings for verbose train vs. test log-likelihood estimation and graphing
    groupings = get_session_groupings(data_metadata, list(data_metadata['groups']), all_keys, hold_out_list)

    # Train ARHMM
    arhmm, loglikes_sample, labels_sample, iter_lls, iter_holls, group_idx = train_model\
    (
        model=arhmm,
        num_iter=config_data['num_iter'],
        ncpus=config_data['ncpus'],
        checkpoint_freq=checkpoint_freq,
        checkpoint_file=checkpoint_file,
        start=itr,
        progress_kwargs=progressbar_kwargs,
        num_frames=nt_frames,
        train_data=train_data,
        val_data=test_data,
        separate_trans=config_data['separate_trans'],
        groups=groupings,
        converge=config_data['converge'],
        tolerance=config_data['tolerance'],
        verbose=config_data['verbose']
    )

    ## Graph training summary
    img_path = graph_modeling_loglikelihoods(config_data, iter_lls, iter_holls, group_idx, dest_file)

    click.echo('Computing likelihoods on each training dataset...')
    # Get training log-likelihoods
    train_ll = get_loglikelihoods(arhmm, train_data, list(data_metadata['groups']), config_data['separate_trans'])

    heldout_ll = []
    # Get held out log-likelihoods
    if config_data['hold_out']:
        click.echo('Computing held out likelihoods with separate transition matrix...')
        heldout_ll = get_loglikelihoods(arhmm, test_data, list(data_metadata['groups']), config_data['separate_trans'])

    loglikes = [loglikes_sample]
    labels = [labels_sample]
    save_parameters = [get_parameters_from_model(arhmm)]

    # if we save the model, don't use copy_model which strips out the data and potentially
    # leaves certain functions useless. We'll want to use in the future (e.g. cross-likes)
    if config_data['e_step']:
        click.echo('Running E step...')
        expected_states = run_e_step(arhmm)

    # TODO:  just compute cross-likes at the end and potentially dump the model (what else
    # would we want the model for hm?), though hard drive space is cheap, recomputing models is not...

    # Pack model data
    export_dict = {
        'loglikes': loglikes,
        'labels': labels,
        'keys': all_keys,
        'heldout_ll': heldout_ll,
        'model_parameters': save_parameters,
        'run_parameters': run_parameters,
        'metadata': data_metadata,
        'model': copy_model(arhmm) if config_data.get('save_model', True) else None,
        'hold_out_list': hold_out_list,
        'train_list': train_list,
        'train_ll': train_ll,
        'expected_states': expected_states if config_data['e_step'] else None
    }

    # Save model
    save_dict(filename=str(dest_file), obj_to_save=export_dict)

    if config_data['verbose']:
        return img_path


def kappa_scan_fit_models_wrapper(input_file, index_file, config_data, output_dir):

    model_name_format = 'model-{}-{}.p '

    data_dict, data_metadata = load_pcs(filename=input_file,
                                        var_name=config_data.get('var_name', 'scores'),
                                        npcs=config_data['npcs'],
                                        load_groups=config_data['load_groups'])

    # get kappa range to scan
    if config_data['min_kappa'] == None or config_data['max_kappa'] == None:

        # Choosing a minimum kappa value (AKA value to begin the scan from)
        # less than the counted number of frames
        if config_data['min_kappa'] == None:
            min_kappa = count_frames(data_dict)/100
            config_data['min_kappa'] = min_kappa

        # get kappa values for each model to train
        if config_data['max_kappa'] == None:
            kappas = [(config_data['min_kappa'] *(10**i)) for i in range(config_data['n_models'])]
        else:
            diff_kappa = config_data['max_kappa'] - config_data['min_kappa']
            kappa_iter = int(diff_kappa / config_data['n_models'])

            kappas = list(range(config_data['min_kappa'], config_data['max_kappa'], kappa_iter))

    else:
        diff_kappa = config_data['max_kappa'] - config_data['min_kappa']
        kappa_iter = int(diff_kappa/config_data['n_models'])

        kappas = list(range(config_data['min_kappa'], config_data['max_kappa'], kappa_iter))

    # get cli command str list
    base_command = f'moseq2-model learn-model {input_file} '

    parameters = f'-i {index_file} --npcs {config_data["npcs"]} -n {config_data["num_iter"]} '

    if config_data['separate_trans']:
        parameters += '--separate-trans '

    if config_data['robust']:
        parameters += '--robust '

    if config_data['e_step']:
        parameters += '--e-step '

    if config_data['hold_out']:
        parameters += f'-h {str(config_data["nfolds"])} '

    if config_data['max_states']:
        parameters += f'-m {config_data["max_states"]} '

    if config_data['converge']:
        parameters += '--converge '

        parameters += f'-t {config_data["tolerance"]} '

    if config_data['cluster_type'] == 'slurm':
        prefix = f'sbatch -c {config_data["ncpus"]} --mem={config_data["memory"]} '
        prefix += f'-p {config_data["partition"]} -t {config_data["wall_time"]} --wrap "'

    commands = []
    for i, k in enumerate(kappas):
        cmd = base_command + os.path.join(output_dir,
              model_name_format.format(str(k), str(i))) + parameters + f'-k {k}'

        if config_data['cluster_type'] == 'slurm':
            cmd = prefix + cmd +'"'
        commands.append(cmd)

    # Display the string
    command_string = '\n'.join(commands)
    print('Listing scan commands...\n')
    print(command_string)

    if not config_data['get_cmd']:
        os.system(command_string)

    return command_string