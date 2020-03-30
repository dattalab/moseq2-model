import os
import click
import random
import warnings
import numpy as np
from cytoolz import pluck
import ruamel.yaml as yaml
from ruamel.yaml import YAML
import matplotlib as plt
from moseq2_model.train.util import whiten_all, whiten_each
from moseq2_model.util import flush_print

def process_indexfile(index, config_data, data_dict, data_metadata):
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
    groups = list(data_metadata['groups'])

    with open(index, 'r') as f:
        index_data = yaml.safe_load(f)
    f.close()

    i_groups, uuids = [], []
    subjectNames, sessionNames = [], []
    for f in index_data['files']:
        if f['uuid'] not in uuids:
            uuids.append(f['uuid'])
            i_groups.append(f['group'])
            subjectNames.append(f['metadata']['SubjectName'])
            sessionNames.append(f['metadata']['SessionName'])

    for i in range(len(subjectNames)):
        print(f'[{i + 1}]', 'Session Name:', sessionNames[i], '; Subject Name:', subjectNames[i], '; group:',
              i_groups[i], '; Key:', uuids[i])

    return index_data, all_keys, groups

def select_data_to_model(index_data):
    use_keys = []
    use_groups = []
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
            print('Group name not found, try again.')

    return use_keys, use_groups

def prepare_model_metadata(data_dict, data_metadata, config_data, alpha, gamma, kappa,
                           nkeys, all_keys, hold_out, nfolds, robust, max_states, separate_trans):

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

    return model_parameters, train_list, hold_out_list

def graph_modeling_loglikelihoods(iter_lls, iter_holls, group_idx, hold_out, verbose, percent_split, nfolds, separate_trans, dest_file):
    if verbose:
        iterations = [i for i in range(len(iter_lls))]
        legend = []
        if separate_trans:
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

        if hold_out:
            img_path = os.path.join(os.path.dirname(dest_file), 'train_heldout_summary.png')
            plt.title('ARHMM Training Summary With '+str(nfolds)+' Folds')
            plt.savefig(img_path)
        else:
            img_path = os.path.join(os.path.dirname(dest_file), f'train_val{percent_split}_summary.png')
            plt.title('ARHMM Training Summary With '+str(percent_split)+'% Train-Val Split')
            plt.savefig(img_path)

    return img_path