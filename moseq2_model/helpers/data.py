import os
import click
import random
import warnings
import numpy as np
import matplotlib.pyplot as plt
from cytoolz import pluck
import ruamel.yaml as yaml
from ruamel.yaml import YAML
from collections import OrderedDict
from moseq2_model.util import flush_print
from sklearn.model_selection import train_test_split
from moseq2_model.train.util import whiten_all, whiten_each

def process_indexfile(index, config_data, data_metadata):
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

        for i in range(len(subjectNames)):
            print(f'[{i + 1}]', 'Session Name:', sessionNames[i], '; Subject Name:', subjectNames[i], '; group:',
                  i_groups[i], '; Key:', uuids[i])
    else:
        index_data = None

    return index_data, data_metadata


def select_data_to_model(index_data, gui=False):
    use_keys = []
    use_groups = []
    if gui:
        groups_to_train = ''
        while (len(use_keys) == 0):
            try:
                groups_to_train = input(
                    "Input comma-separated names of the groups to model. Empty string to model all the sessions/groups in the index file.")
                if ',' in groups_to_train:
                    tmp_g = groups_to_train.split(',')
                    for g in tmp_g:
                        g = g.strip()
                        for f in index_data['files']:
                            if f['group'] == g:
                                if f['uuid'] not in use_keys:
                                    use_keys.append(f['uuid'])
                                    use_groups.append(g)
                elif len(groups_to_train) == 0:
                    for f in index_data['files']:
                        use_keys.append(f['uuid'])
                        use_groups.append(f['group'])
                else:
                    for f in index_data['files']:
                        if f['group'] == groups_to_train:
                            if f['uuid'] not in use_keys:
                                use_keys.append(f['uuid'])
                                use_groups.append(groups_to_train)
            except:
                if len(groups_to_train) == 0:
                    for f in index_data['files']:
                        use_keys.append(f['uuid'])
                        use_groups.append(f['group'])
                    all_keys = use_keys
                    groups = use_groups
                    break
                print('Group name not found, try again.')

            all_keys = use_keys
            groups = use_groups
    else:
        for f in index_data['files']:
            use_keys.append(f['uuid'])
            use_groups.append(f['group'])
        all_keys = use_keys
        groups = use_groups

    return all_keys, groups

def prepare_model_metadata(data_dict, data_metadata, config_data, nkeys, all_keys):
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
        'separate_trans': config_data['separate_trans']
    }

    if config_data['separate_trans']:
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

    return config_data, data_dict, model_parameters, train_list, hold_out_list

def get_heldout_data_splits(all_keys, data_dict, train_list, hold_out_list):
    train_data = OrderedDict((i, data_dict[i]) for i in all_keys if i in train_list)
    test_data = OrderedDict((i, data_dict[i]) for i in all_keys if i in hold_out_list)
    train_list = list(train_data.keys())
    hold_out_list = list(test_data.keys())
    nt_frames = [len(v) for v in train_data.values()]

    return train_list, train_data, hold_out_list, test_data, nt_frames

def get_training_data_splits(config_data, data_dict):

    train_data = data_dict
    train_list = list(data_dict.keys())
    test_data = None
    hold_out_list = None

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

    return train_list, train_data, training_data, hold_out_list, validation_data, nt_frames

def graph_modeling_loglikelihoods(config_data, iter_lls, iter_holls, group_idx, dest_file):
    if config_data['verbose']:
        iterations = [i for i in range(len(iter_lls))]
        legend = []
        if config_data['separate_trans']:
            for i, g in enumerate(group_idx):
                lw = 10 - 8 * i / len(iter_lls[0])
                ls = ['-', '--', '-.', ':'][i % 4]

                plt.plot(iterations, np.asarray(iter_lls)[:, i], linestyle=ls, linewidth=lw)
                legend.append(f'train: {g} LL')

            for i, g in enumerate(group_idx):
                lw = 5 - 3 * i / len(iter_holls[0])
                ls = ['-', '--', '-.', ':'][i % 4]

                plt.plot(iterations, np.asarray(iter_holls)[:, i], linestyle=ls, linewidth=lw)
                legend.append(f'val: {g} LL')
        else:
            for i, g in enumerate(group_idx):
                lw = 10 - 8 * i / len(iter_lls)
                ls = ['-', '--', '-.', ':'][i % 4]

                plt.plot(iterations, np.asarray(iter_lls), linestyle=ls, linewidth=lw)
                legend.append(f'train: {g} LL')

            for i, g in enumerate(group_idx):
                lw = 5 - 3 * i / len(iter_holls)
                ls = ['-', '--', '-.', ':'][i % 4]
                try:
                    plt.plot(iterations, np.asarray(iter_holls)[:, i], linestyle=ls, linewidth=lw)
                    legend.append(f'val: {g} LL')
                except:
                    plt.plot(iterations, np.asarray(iter_holls), linestyle=ls, linewidth=lw)
                    legend.append(f'val: {g} LL')
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

