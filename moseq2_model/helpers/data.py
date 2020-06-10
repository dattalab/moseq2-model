import os
import click
import random
import warnings
import numpy as np
from cytoolz import pluck
import ruamel.yaml as yaml
from ruamel.yaml import YAML
import matplotlib.pyplot as plt
from collections import OrderedDict
from moseq2_model.util import flush_print
from sklearn.model_selection import train_test_split
from moseq2_model.train.util import whiten_all, whiten_each

def process_indexfile(index, config_data, data_metadata):
    '''
    Reads index file (if it exists) and returns dictionaries containing metadata in the index file.
    The data_metadata will also be updated with the information read from the index file

    Parameters
    ----------
    index (str): path to index file.
    config_data (dict): dictionary containing all modeling parameters.
    data_metadata (dict): loaded metadata containing uuid and group information.

    Returns
    -------
    index_data (dict): dictionary containing data contained in the index file.
    data_metadata (dict): updated metadata dictionary.
    '''

    # if we have an index file, strip out the groups, match to the scores uuids
    if os.path.exists(str(index)):
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

        with open(index, 'r') as f:
            index_data = yaml.safe_load(f)
        f.close()

        i_groups, uuids = [], []
        subjectNames, sessionNames = [], []
        for f in index_data['files']:
            if f['uuid'] not in uuids:
                uuids.append(f['uuid'])
                i_groups.append(f['group'])
                try:
                    subjectNames.append(f['metadata']['SubjectName'])
                    sessionNames.append(f['metadata']['SessionName'])
                except:
                    f['metadata'] = {}
                    f['metadata']['SubjectName'] = 'default'
                    f['metadata']['SessionName'] = 'default'
                    subjectNames.append(f['metadata']['SubjectName'])
                    sessionNames.append(f['metadata']['SessionName'])

        with open(index, 'w') as f:
            yaml.safe_dump(index_data, f)
        f.close()

        if config_data.get('select_groups', False):
            for i in range(len(subjectNames)):
                print(f'[{i + 1}]', 'Session Name:', sessionNames[i], '; Subject Name:', subjectNames[i], '; group:',
                      i_groups[i], '; Key:', uuids[i])
    else:
        index_data = None

    return index_data, data_metadata


def select_data_to_model(index_data, select_groups=False):
    '''
    GUI: Prompts user to select data to model via the data uuids/groups and paths located in the index file.
    CLI: Selects all data from index file.

    Parameters
    ----------
    index_data (dict): loaded dictionary from index file
    gui (bool): indicates prompting user input

    Returns
    -------
    all_keys (list): list of uuids to model
    groups (list): list of groups to model
    '''

    use_keys = []
    use_groups = []
    if select_groups:
        while(len(use_groups) == 0):
            groups_to_train = input(
                "Input comma/space-separated names of the groups to model. Empty string to model all the sessions/groups in the index file.")
            if ',' in groups_to_train:
                sel_groups = [g.strip() for g in groups_to_train.split(',')]
                use_keys = [f['uuid'] for f in index_data['files'] if f['group'] in sel_groups]
                use_groups = [f['group'] for f in index_data['files'] if f['uuid'] in use_keys]
            elif len(groups_to_train) > 0:
                sel_groups = [g for g in groups_to_train.split(' ')]
                use_keys = [f['uuid'] for f in index_data['files'] if f['group'] in sel_groups]
                use_groups = [f['group'] for f in index_data['files'] if f['uuid'] in use_keys]
            else:
                for f in index_data['files']:
                    use_keys.append(f['uuid'])
                    use_groups.append(f['group'])
    else:
        for f in index_data['files']:
            use_keys.append(f['uuid'])
            use_groups.append(f['group'])

    all_keys = use_keys
    groups = use_groups

    return all_keys, groups

def prepare_model_metadata(data_dict, data_metadata, config_data, nkeys, all_keys):
    '''
    Sets model training metadata parameters, whitens data,
    if hold_out is True, will split data and return list of heldout keys,
    and updates all dictionaries.

    Parameters
    ----------
    data_dict (OrderedDict): loaded data dictionary.
    data_metadata (OrderedDict): loaded metadata dictionary.
    config_data (dict): dictionary containing all modeling parameters.
    nkeys (int): total amount of keys being modeled.
    all_keys (list): list of keys being modeled.

    Returns
    -------
    config_data (dict): updated dictionary containing all modeling parameters.
    data_dict (OrderedDict): update data dictionary.
    model_parameters (dict): dictionary of pre-selected model parameters
    train_list (list): list of keys included in training list.
    hold_out_list (list): heldout list of keys (if hold_out == True)
    '''

    if config_data['kappa'] is None:
        total_frames = 0
        for v in data_dict.values():
            idx = (~np.isnan(v)).all(axis=1)
            total_frames += np.sum(idx)
        flush_print(f'Setting kappa to the number of frames: {total_frames}')
        config_data['kappa'] = total_frames

    if config_data['hold_out'] and nkeys >= config_data['nfolds']:
        click.echo(f"Will hold out 1 fold of {config_data['nfolds']}")

        if config_data['hold_out_seed'] >= 0:
            click.echo(f"Settings random seed to {config_data['hold_out_seed']}")
            splits = np.array_split(random.Random(config_data['hold_out_seed']).sample(list(range(nkeys)), nkeys),
                                    config_data['nfolds'])
        else:
            warnings.warn("Random seed not set, will choose a different test set each time this is run...")
            splits = np.array_split(random.sample(list(range(nkeys)), nkeys), config_data['nfolds'])

        hold_out_list = [all_keys[k] for k in splits[0].astype('int').tolist()]
        train_list = [k for k in all_keys if k not in hold_out_list]
        click.echo("Holding out " + str(hold_out_list))
        click.echo("Training on " + str(train_list))
    else:
        config_data['hold_out'] = False
        hold_out_list = None
        train_list = all_keys
        test_data = None

    if config_data['ncpus'] > len(train_list):
        ncpus = len(train_list)
        config_data['ncpus'] = ncpus
        warnings.warn(f'Setting ncpus to {nkeys}, ncpus must be <= nkeys in dataset, {len(train_list)}')

    # use a list of dicts, with everything formatted ready to go
    model_parameters = {
        'gamma': config_data['gamma'],
        'alpha': config_data['alpha'],
        'kappa': config_data['kappa'],
        'nlags': config_data['nlags'],
        'robust': config_data['robust'],
        'max_states': config_data['max_states'],
        'separate_trans': config_data['separate_trans'],
        'groups': None
    }

    if config_data['separate_trans']:
        model_parameters['groups'] = data_metadata['groups']

    if config_data['whiten'][0].lower() == 'a':
        click.echo('Whitening the training data using the whiten_all function')
        data_dict = whiten_all(data_dict)
    elif config_data['whiten'][0].lower() == 'e':
        click.echo('Whitening the training data using the whiten_each function')
        data_dict = whiten_each(data_dict)
    else:
        click.echo('Not whitening the data')

    if config_data['noise_level'] > 0:
        click.echo(f'Using {config_data["noise_level"]} STD AWGN.')
        for k, v in data_dict.items():
            data_dict[k] = v + np.random.randn(*v.shape) * config_data['noise_level']

    return config_data, data_dict, model_parameters, train_list, hold_out_list

def get_heldout_data_splits(all_keys, data_dict, train_list, hold_out_list):
    '''
    Split data based on held out keys.

    Parameters
    ----------
    all_keys (list): list of all keys included in the model.
    data_dict (OrderedDict): dictionary of all PC scores included in the model
    train_list (list): list of keys included in the training data
    hold_out_list (list): list of keys included in the held out data

    Returns
    -------
    train_list (list):  list of keys included in the training data.
    train_data (OrderedDict): dictionary of uuid to PC score key-value pairs for uuids in train_list
    hold_out_list (list): list of keys included in the held out data.
    test_data (OrderedDict): dictionary of uuids to PC score key-value pairs for uuids in hold_out_list.
    nt_frames (list): list of the number of frames in each session in train_data
    '''

    train_data = OrderedDict((i, data_dict[i]) for i in all_keys if i in train_list)
    test_data = OrderedDict((i, data_dict[i]) for i in all_keys if i in hold_out_list)
    hold_out_list = list(test_data.keys())
    nt_frames = [len(v) for v in train_data.values()]

    return train_data, hold_out_list, test_data, nt_frames

def get_training_data_splits(config_data, data_dict):
    '''
    Split data using sklearn train_test_split along all keys.

    Parameters
    ----------
    config_data (dict): dictionary containing percentage split parameter. (autogenerated in GUI AND CLI)
    data_dict (OrderedDict): dict of uuid-PC Score key-value pairs for all data included in the model.

    Returns
    -------
    train_list (list): list of all the keys included in the model.
    train_data (OrderedDict): all the of the key-value pairs included in the model.
    training_data (OrderedDict): the split percentage of the training data.
    hold_out_list (list): None
    validation_data (OrderedDict): the split percentage of the validation data
    nt_frames (list): list of length of each session in the split training data.
    '''

    train_data = data_dict

    training_data = OrderedDict()
    validation_data = OrderedDict()

    nt_frames = []
    nv_frames = []

    for k, v in train_data.items():
        # train values
        training_X, testing_X = train_test_split(v, test_size=config_data['percent_split'] / 100, shuffle=False, random_state=0)
        training_data[k] = training_X
        nt_frames.append(training_data[k].shape[0])

        # validation values
        validation_data[k] = testing_X
        nv_frames.append(validation_data[k].shape[0])

    return train_data, training_data, validation_data, nt_frames

def graph_helper(groups, lls, legend, iterations, ll_type='train', sep_trans=False):
    '''
    Helper function to plot the training and validation log-likelihoods
     over the each model training iteration.

    Parameters
    ----------
    groups (list): list of group names that the model was trained on.
    lls (list): list of log-likelihoods over each iteration.
    legend (list): list of legend labels for each group's log-likelihoods curve.
    iterations (list): range() generated list indicated x-axis length.
    ll_type (str): string to indicate (in the legend) whether plotting training or validation curves.
    sep_trans (bool): indicates whether there is more than one set on log-likelihoods.

    Returns
    -------
    None
    '''

    if sep_trans: lls = lls[0]
    for i, g in enumerate(groups):
        lw = 10 - 8 * i / len(lls)
        ls = ['-', '--', '-.', ':'][i % 4]
        try:
            plt.plot(iterations, np.asarray(lls)[:, i], linestyle=ls, linewidth=lw)
        except:
            plt.plot(iterations, np.asarray(lls), linestyle=ls, linewidth=lw)
        legend.append(f'{ll_type}: {g} LL')

def graph_modeling_loglikelihoods(config_data, iter_lls, iter_holls, group_idx, dest_file):
    '''
    Graphs model training performance progress throughout modeling.
    Will only run if verbose == True

    Parameters
    ----------
    config_data (dict): dictionary of model training parameters.
    iter_lls (list): list of training log-likelihoods over each iteration
    iter_holls (list): list of held out log-likelihoods over each iteration
    group_idx (list): list of groups included in the modeling.
    dest_file (str): path to the model.

    Returns
    -------
    img_path (str): path to saved graph.
    '''

    if config_data['verbose']:
        iterations = [i for i in range(len(iter_lls))]
        legend = []
        if config_data['separate_trans']:
            graph_helper(group_idx, iter_lls, legend, iterations, sep_trans=True)
            graph_helper(group_idx, iter_holls, legend, iterations, ll_type='val', sep_trans=True)
        else:
            graph_helper(group_idx, iter_lls, legend, iterations)
            graph_helper(group_idx, iter_lls, legend, iterations, ll_type='val')

        plt.legend(legend)

        plt.ylabel('Average Syllable Log-Likelihood')
        plt.xlabel('Iterations')

        if config_data['hold_out']:
            img_path = os.path.join(os.path.dirname(dest_file), 'train_heldout_summary.png')
            plt.title('ARHMM Training Summary With ' + str(config_data['nfolds']) + ' Folds')
            plt.savefig(img_path)
        else:
            img_path = os.path.join(os.path.dirname(dest_file), f'train_val{config_data["percent_split"]}_summary.png')
            plt.title('ARHMM Training Summary With ' + str(config_data["percent_split"]) + '% Train-Val Split')
            plt.savefig(img_path)

        return img_path