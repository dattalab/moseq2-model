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
from moseq2_model.util import (save_dict, load_pcs, get_parameters_from_model, copy_model, get_scan_range_kappas,
                               create_command_strings, get_current_model, get_loglikelihoods, get_session_groupings)
from moseq2_model.helpers.data import (process_indexfile, select_data_to_model, prepare_model_metadata,
                                       graph_modeling_loglikelihoods, get_heldout_data_splits, get_training_data_splits)

def learn_model_wrapper(input_file, dest_file, config_data):
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

    dest_file = realpath(dest_file)

    if not exists(dirname(dest_file)):
        os.makedirs(dirname(dest_file))

    if not os.access(dirname(dest_file), os.W_OK):
        raise IOError('Output directory is not writable.')

    # Handle checkpoint parameters
    checkpoint_path = join(dirname(dest_file), 'checkpoints/')
    checkpoint_freq = config_data.get('checkpoint_freq', -1)

    if checkpoint_freq < 0:
        checkpoint_freq = None
    elif not exists(checkpoint_path):
        os.makedirs(checkpoint_path)

    click.echo("Entering modeling training")

    run_parameters = deepcopy(config_data)

    # Get session PC scores and session metadata dicts
    data_dict, data_metadata = load_pcs(filename=input_file,
                                        var_name=config_data.get('var_name', 'scores'),
                                        npcs=config_data['npcs'],
                                        load_groups=config_data['load_groups'])

    # Parse index file and update metadata information; namely groups
    index_data, data_metadata = process_indexfile(config_data.get('index', None), config_data, data_metadata, config_data['default_group'])

    # Get all training session uuids
    all_keys = list(data_dict)
    groups = list(data_metadata['groups'])

    # Get keys to include in training set
    select_groups = config_data.get('select_groups', False)
    if index_data != None:
        all_keys, groups = select_data_to_model(index_data, select_groups)
        data_metadata['groups'] = groups
        data_metadata['uuids'] = all_keys

    nkeys = len(data_dict)

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
    checkpoint_file = basename(dest_file).replace('.p', '')
    all_checkpoints = glob.glob(join(checkpoint_path, f'{checkpoint_file}*.arhmm'))

    # Instantiate model; either anew or from previously saved checkpoint
    arhmm, itr = get_current_model(config_data['use_checkpoint'], all_checkpoints, train_data, model_parameters)

    # Pack progress bar keyword arguments
    progressbar_kwargs = {
        'total': config_data['num_iter'],
        'file': sys.stdout,
        'leave': False,
        'disable': not config_data['progressbar'],
        'initial': itr
    }

    # Get data groupings for verbose train vs. test log-likelihood estimation and graphing
    if hold_out_list != None and groups != None:
        groupings = get_session_groupings(data_metadata, all_keys, hold_out_list)
    else:
        groupings = None

    # Train ARHMM
    arhmm, loglikes_sample, labels_sample, iter_lls, iter_holls, group_idx = train_model(
        model=arhmm,
        num_iter=config_data['num_iter'],
        ncpus=config_data['ncpus'],
        checkpoint_freq=checkpoint_freq,
        checkpoint_file=join(checkpoint_path, checkpoint_file),
        start=itr,
        progress_kwargs=progressbar_kwargs,
        num_frames=nt_frames,
        train_data=train_data,
        val_data=test_data,
        separate_trans=config_data['separate_trans'],
        groups=groupings,
        check_every=config_data['check_every'],
        verbose=config_data['verbose']
    )

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

    # TODO: just compute cross-likes at the end and potentially dump the model (what else
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

    if config_data['verbose'] and len(iter_lls) > 0:
        img_path = graph_modeling_loglikelihoods(config_data, iter_lls, iter_holls, group_idx, dest_file)
        return img_path


def kappa_scan_fit_models_wrapper(input_file, config_data, output_dir):
    '''
    Wrapper function that spools multiple model training commands for different kappa values within a
     given range. (Either n models with kappa values equally spaced between a min and max value, or
     choosing n kappa values ranging in factors of 10 starting from nframes/100 for n=number of models).

    Parameters
    ----------
    input_file (str): Path to PCA Scores
    config_data (dict): Dict containing model training parameters
    output_dir (str): Path to output directory to save trained models

    Returns
    -------
    command_string (str): CLI command string to sequential
     (or parallel in case of cluster-type=='slurm') model training commands.
    '''

    data_dict, _ = load_pcs(filename=input_file, var_name=config_data.get('var_name', 'scores'),
                            npcs=config_data['npcs'], load_groups=config_data['load_groups'])

    # Get list of kappa values for spooling models
    kappas = get_scan_range_kappas(data_dict, config_data)

    # Get model training command strings
    command_string = create_command_strings(input_file, output_dir, config_data, kappas)

    # Ensure output directory exists
    if not exists(dirname(config_data['out_script'])):
        os.makedirs(dirname(config_data['out_script']))

    # Write command string to file
    with open(config_data['out_script'], 'w') as f:
        f.write(command_string)
    print('Commands saved to:', config_data['out_script'])

    if config_data['get_cmd']:
        # Display the command string
        print('Listing kappa scan commands...\n')
        print(command_string)
    if config_data['run_cmd']:
        # Or run the kappa scan
        print('Running kappa scan commands')
        os.system(command_string)

    return command_string