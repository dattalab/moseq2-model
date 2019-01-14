import click
import os
import sys
import shutil
import random
import warnings
import numpy as np
from copy import deepcopy
from collections import OrderedDict
from moseq2_model.train.models import ARHMM
from moseq2_model.train.util import train_model, whiten_all, whiten_each
from moseq2_model.util import save_dict, load_pcs, get_parameters_from_model, copy_model, load_arhmm_checkpoint

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

    data_dict, data_metadata = load_pcs(filename=input_file, var_name=var_name,
                                        npcs=10, load_groups=False)
    total_frames = 0
    for v in data_dict.values():
        idx = (~np.isnan(v)).all(axis=1)
        total_frames += np.sum(idx)

    print('Total frames: {}'.format(total_frames))


# this is the entry point for learning models over Kubernetes, expose all
# parameters we could/would possibly scan over
@cli.command(name="learn-model")
@click.argument("input_file", type=click.Path(exists=True))
@click.argument("dest_file", type=click.Path(file_okay=True, writable=True))
@click.option("--hold-out", "-h", type=bool, default=False, is_flag=True,
              help="Hold out one fold (set by nfolds) for computing heldout likelihood")
@click.option("--hold-out-seed", type=int, default=-1,
              help="Random seed for holding out data (set for reproducibility)")
@click.option("--nfolds", type=int, default=5, help="Number of folds for split")
@click.option("--ncpus", "-c", type=int, default=0, help="Number of cores to use for resampling")
@click.option("--num-iter", "-n", type=int, default=100, help="Number of times to resample model")
@click.option("--restarts", "-r", type=int, default=1, help="Number of restarts for model")
@click.option("--var-name", type=str, default='scores', help="Variable name in input file with PCs")
@click.option("--save-every", "-s", type=int, default=-1,
              help="Increment to save labels and model object (-1 for just last)")
@click.option("--save-model", is_flag=True, help="Save model object")
@click.option("--max-states", "-m", type=int, default=100, help="Maximum number of states")
@click.option("--model-progress", "-p", type=bool, default=True, help="Show model progress")
@click.option("--save-model-progress", type=int, default=None, help='Save the model object after this many iterations')
@click.option("--npcs", type=int, default=10, help="Number of PCs to use")
@click.option("--whiten", "-w", type=str, default='all', help="Whiten (e)each (a)ll or (n)o whitening")
@click.option("--kappa", "-k", type=float, default=None, help="Kappa")
@click.option("--gamma", "-g", type=float, default=1e3, help="Gamma")
@click.option("--alpha", "-g", type=float, default=5.7, help="Alpha")
@click.option("--nu", type=float, default=4, help="Nu (only applicable if robust set to true)")
@click.option("--nlags", type=int, default=3, help="Number of lags to use")
@click.option("--separate-trans", is_flag=True, help="Use separate transition matrix per group")
@click.option("--robust", is_flag=True, help="Use tAR model")
def learn_model(input_file, dest_file, hold_out, hold_out_seed, nfolds, ncpus,
                num_iter, restarts, var_name,
                save_every, save_model, max_states, model_progress, npcs, whiten,
                kappa, gamma, alpha, nu, nlags, separate_trans, robust, save_model_progress):

    # TODO: graceful handling of extra parameters:  orchestrating this fails catastrophically if we pass
    # an extra option, just flag it to the user and ignore

    if not(os.path.dirname(dest_file)):
        dest_file = os.path.join('./', dest_file)

    if not os.access(os.path.dirname(dest_file), os.W_OK):
        raise IOError('Output directory is not writable.')

    if save_every < 0:
        click.echo("Will only save the last iteration of the model")
        save_every = num_iter+1

    click.echo("Entering modeling training")

    run_parameters = deepcopy(locals())
    data_dict, data_metadata = load_pcs(filename=input_file, var_name=var_name,
                                        npcs=npcs, load_groups=separate_trans)
    all_keys = list(data_dict.keys())
    nkeys = len(all_keys)
    compute_heldouts = False

    if kappa is None:
        total_frames = 0
        for v in data_dict.values():
            idx = (~np.isnan(v)).all(axis=1)
            total_frames += np.sum(idx)

        print('Setting kappa to the number of frames: {}'.format(total_frames))
        kappa = total_frames

    if hold_out and nkeys >= nfolds:
        click.echo("Will hold out 1 fold of "+str(nfolds))

        if hold_out_seed >= 0:
            click.echo("Settings random seed to "+str(hold_out_seed))
            splits = np.array_split(random.Random(hold_out_seed).sample(list(range(nkeys)), nkeys), nfolds)
        else:
            warnings.warn("Random seed not set, will choose a different test set each time this is run...")
            splits = np.array_split(random.sample(list(range(nkeys)), nkeys), nfolds)

        hold_out_list = [all_keys[k] for k in splits[0].astype('int').tolist()]
        train_list = [k for k in all_keys if k not in hold_out_list]
        click.echo("Holding out "+str(hold_out_list))
        click.echo("Training on "+str(train_list))
        compute_heldouts = True
    else:
        hold_out_list = None
        train_list = all_keys

    if ncpus > len(train_list):
        warnings.warn('Setting ncpus to {}, ncpus must be <= nkeys in dataset, {}'.format(nkeys, len(train_list)))
        ncpus = len(train_list)

    # use a list of dicts, with everything formatted ready to go

    model_parameters = {
        'gamma': gamma,
        'alpha': alpha,
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

    if compute_heldouts:
        train_data = OrderedDict((i, data_dict[i]) for i in all_keys if i in train_list)
        test_data = OrderedDict((i, data_dict[i]) for i in all_keys if i in hold_out_list)
        train_list = list(train_data.keys())
        hold_out_list = list(test_data.keys())
    else:
        train_data = data_dict
        test_data = None
        train_list = list(data_dict.keys())
        test_list = None

    loglikes = []
    labels = []
    heldout_ll = []
    save_parameters = []

    checkpoint_file = os.path.splitext(dest_file)[0] + '-checkpoint.arhmm'


    for i in range(restarts):
        # look for model checkpoint
        if os.path.exists(checkpoint_file):
            print('Loading checkpoint')
            try:
                checkpoint = load_arhmm_checkpoint(checkpoint_file)
                if os.path.exists(checkpoint_file + '.1'):
                    os.remove(checkpoint_file + '.1')
                shutil.move(checkpoint_file, checkpoint_file + '.1')
            except ValueError:
                print('Loading original checkpoint failed, checking backups')
                os.remove(checkpoint_file)
                checkpoint = load_arhmm_checkpoint(checkpoint_file + '.1')
            arhmm = checkpoint.pop('model')
            print('On iteration', checkpoint['iter'])
        else:
            arhmm = ARHMM(data_dict=train_data, **model_parameters)
            checkpoint = dict(iter=0)
        arhmm, loglikes_sample, labels_sample = train_model(
            model=arhmm,
            save_every=save_every,
            num_iter=num_iter,
            cli=True,
            leave=False,
            disable=not model_progress,
            total=num_iter*restarts,
            initial=i*num_iter,
            ncpus=ncpus,
            file=sys.stdout,
            save_progress=save_model_progress,
            filename=dest_file, **checkpoint
        )

        if test_data and separate_trans:
            click.echo("Computing held out likelihoods with separate transition matrix...")
            [heldout_ll.append(arhmm.log_likelihood(v, group_id=data_metadata['groups'][i]))
                for i, (k, v) in enumerate(test_data.items())]
        elif test_data:
            click.echo("Computing held out likelihoods...")
            [heldout_ll.append(arhmm.log_likelihood(v)) for k, v in test_data.items()]

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
        'keys': all_keys,
        'heldout_ll': heldout_ll,
        'model_parameters': save_parameters,
        'run_parameters': run_parameters,
        'metadata': data_metadata,
        'model': save_model,
        'hold_out_list': hold_out_list,
        'train_list': train_list
        }

    save_dict(filename=dest_file, obj_to_save=export_dict)


if __name__ == '__main__':
    cli()
