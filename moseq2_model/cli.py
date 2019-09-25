import click
import os
import sys
import random
import warnings
import numpy as np
from pathlib import Path
from copy import deepcopy
from moseq2_model.train.util import train_model, whiten_all, whiten_each, run_e_step
from moseq2_model.util import (save_dict, load_pcs, get_parameters_from_model, copy_model,
                               load_arhmm_checkpoint, flush_print)
from ruamel.yaml import YAML
from collections import OrderedDict
from moseq2_model.train.models import ARHMM
from moseq2_model.train.util import train_model, whiten_all, whiten_each, run_e_step
from moseq2_model.util import (save_dict, load_pcs, get_parameters_from_model, copy_model,
                               load_arhmm_checkpoint, flush_print)

orig_init = click.core.Option.__init__


def new_init(self, *args, **kwargs):
    orig_init(self, *args, **kwargs)
    self.show_default = True


click.core.Option.__init__ = new_init


@click.group()
def cli():
    pass


@cli.command(name='count-frames')
@click.argument("input_file", type=click.Path(exists=True))
@click.option("--var-name", type=str, default='scores', help="Variable name in input file with PCs")
def count_frames(input_file, var_name):

    data_dict, _ = load_pcs(filename=input_file, var_name=var_name,
                            npcs=10, load_groups=False)
    total_frames = 0
    for v in data_dict.values():
        idx = (~np.isnan(v)).all(axis=1)
        total_frames += np.sum(idx)

    flush_print('Total frames:', total_frames)


# this is the entry point for learning models over Kubernetes, expose all
# parameters we could/would possibly scan over
@cli.command(name="learn-model")
@click.argument("input_file", type=click.Path(exists=True))
@click.argument("dest_file", type=click.Path(file_okay=True, writable=True, resolve_path=True))
@click.option("--hold-out", "-h", type=bool, default=False, is_flag=True,
              help="Hold out one fold (set by nfolds) for computing heldout likelihood")
@click.option("--hold-out-seed", type=int, default=-1,
              help="Random seed for holding out data (set for reproducibility)")
@click.option("--nfolds", type=int, default=5, help="Number of folds for split")
@click.option("--ncpus", "-c", type=int, default=0, help="Number of cores to use for resampling")
@click.option("--num-iter", "-n", type=int, default=100, help="Number of times to resample model")
@click.option("--var-name", type=str, default='scores', help="Variable name in input file with PCs")
@click.option('--e-step', is_flag=True, help="Compute the expected state values for each animal")
@click.option("--save-every", "-s", type=int, default=-1,
              help="Increment to save labels and model object (-1 for just last)")
@click.option("--save-model", is_flag=True, help="Save model object at the end of training")
@click.option("--max-states", "-m", type=int, default=100, help="Maximum number of states")
@click.option("--npcs", type=int, default=10, help="Number of PCs to use")
@click.option("--whiten", "-w", type=str, default='all', help="Whiten (e)each (a)ll or (n)o whitening")
@click.option("--progressbar", "-p", type=bool, default=True, help="Show model progress")
@click.option("--kappa", "-k", type=float, default=None, help="Kappa")
@click.option("--gamma", "-g", type=float, default=1e3, help="Gamma")
@click.option("--alpha", "-g", type=float, default=5.7, help="Alpha")
@click.option("--noise-level", type=float, default=0, help="Additive white gaussian noise for regularization" )
@click.option("--nlags", type=int, default=3, help="Number of lags to use")
@click.option("--separate-trans", is_flag=True, help="Use separate transition matrix per group")
@click.option("--robust", is_flag=True, help="Use tAR model")
@click.option("--checkpoint-freq", type=int, default=-1, help='checkpoint the training after N iterations')
@click.option("--index", "-i", type=click.Path(), default="", help="Path to moseq2-index.yaml for group definitions (used only with the separate-trans flag)")
@click.option("--default-group", type=str, default="n/a", help="Default group to use for separate-trans")
@click.option('--e-step', is_flag=True, help="Compute the expected state values for each animal")
def learn_model(input_file, dest_file, hold_out, hold_out_seed, nfolds, ncpus,
                num_iter, var_name, e_step,
                save_every, save_model, max_states, npcs, whiten, progressbar,
                kappa, gamma, alpha, noise_level, nlags, separate_trans, robust,
                checkpoint_freq, index, default_group):

    # TODO: graceful handling of extra parameters:  orchestrating this fails catastrophically if we pass
    # an extra option, just flag it to the user and ignore
    dest_file = os.path.realpath(dest_file)

    # if not os.path.dirname(dest_file):
    #     dest_file = os.path.join('./', dest_file)

    if not os.access(os.path.dirname(dest_file), os.W_OK):
        raise IOError('Output directory is not writable.')

    if save_every < 0:
        click.echo("Will only save the last iteration of the model")
        save_every = num_iter + 1

    if checkpoint_freq < 0:
        checkpoint_freq = num_iter + 1

    click.echo("Entering modeling training")

    run_parameters = deepcopy(locals())
    data_dict, data_metadata = load_pcs(filename=input_file,
                                        var_name=var_name,
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
                data_metadata["groups"].append(default_group)

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

        if hold_out_seed >= 0:
            click.echo(f"Settings random seed to {hold_out_seed}")
            splits = np.array_split(random.Random(hold_out_seed).sample(list(range(nkeys)), nkeys), nfolds)
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

    if ncpus > len(train_list):
        ncpus = len(train_list)
        warnings.warn(f'Setting ncpus to {nkeys}, ncpus must be <= nkeys in dataset, {len(train_list)}')

    # use a list of dicts, with everything formatted ready to go
    model_parameters = {
        'gamma': gamma,
        'alpha': alpha,
        'kappa': kappa,
        'nlags': nlags,
        'separate_trans': separate_trans,
        'robust': robust,
        'max_states': max_states,
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

    if noise_level > 0:
        click.echo('Using {} STD AWGN'.format(noise_level))
        for k, v in data_dict.items():
            data_dict[k] = v + np.random.randn(*v.shape) * noise_level

    '''
    if compute_heldouts:
        train_data = OrderedDict((i, data_dict[i]) for i in all_keys if i in train_list)
        test_data = OrderedDict((i, data_dict[i]) for i in all_keys if i in hold_out_list)
        train_list = list(train_data.keys())
        hold_out_list = list(test_data.keys())
    else:
        train_data = data_dict
        train_list = list(data_dict.keys())
        test_list = None
    '''

    if hold_out:
        train_data = OrderedDict((i, data_dict[i]) for i in all_keys if i in train_list)
        test_data = OrderedDict((i, data_dict[i]) for i in all_keys if i in hold_out_list)
        train_list = list(train_data.keys())
        hold_out_list = list(test_data.keys())
    else:
        train_data = data_dict
        train_list = list(data_dict.keys())

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
        'disable': not progressbar,
        'initial': itr
    }

    arhmm, loglikes_sample, labels_sample = train_model(
        model=arhmm,
        save_every=save_every,
        num_iter=num_iter,
        ncpus=ncpus,
        checkpoint_freq=checkpoint_freq,
        save_file=resample_save_file,
        chkpt_file=checkpoint_file,
        start=itr,
        progress_kwargs=progressbar_kwargs,
    )

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
    if e_step:
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
        'model': copy_model(arhmm) if save_model else None,
        'hold_out_list': hold_out_list,
        'train_list': train_list,
        'train_ll': train_ll
    }

    if e_step:
        export_dict['expected_states'] = expected_states

    save_dict(filename=str(dest_file), obj_to_save=export_dict)

if __name__ == '__main__':
    cli()
