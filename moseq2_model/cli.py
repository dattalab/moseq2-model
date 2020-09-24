'''
CLI front-end operations. This module contains all the functionality and configurable parameters
users can alter to most accurately process their data.
'''

import click
import numpy as np
from moseq2_model.util import load_pcs
from moseq2_model.train.models import flush_print
from moseq2_model.helpers.wrappers import learn_model_wrapper, kappa_scan_fit_models_wrapper

orig_init = click.core.Option.__init__


def new_init(self, *args, **kwargs):
    orig_init(self, *args, **kwargs)
    self.show_default = True


click.core.Option.__init__ = new_init


@click.group()
@click.version_option()
def cli():
    pass


@cli.command(name='count-frames', help="Counts number of frames in given h5 file (pca_scores)")
@click.argument("input_file", type=click.Path(exists=True))
@click.option("--var-name", type=str, default='scores', help="Variable name in input file with PCs")
def count_frames(input_file, var_name):

    data_dict, _ = load_pcs(filename=input_file, var_name=var_name,
                            npcs=10, load_groups=True)
    total_frames = 0
    for v in data_dict.values():
        idx = (~np.isnan(v)).all(axis=1)
        total_frames += np.sum(idx)

    flush_print('Total frames:', total_frames)


def modeling_parameters(function):

    function = click.option("--hold-out", "-h", type=bool, default=False, is_flag=True,
              help="Hold out one fold (set by nfolds) for computing heldout likelihood")(function)
    function = click.option("--hold-out-seed", type=int, default=-1,
              help="Random seed for holding out data (set for reproducibility)")(function)
    function = click.option("--nfolds", type=int, default=5, help="Number of folds for split")(function)
    function = click.option("--ncpus", "-c", type=int, default=0, help="Number of cores to use for resampling")(function)
    function = click.option("--num-iter", "-n", type=int, default=100, help="Number of times to resample model")(function)
    function = click.option("--var-name", type=str, default='scores', help="Variable name in input file with PCs")(function)
    function = click.option('--e-step', is_flag=True, help="Compute the expected state values for each animal")(function)
    function = click.option("--save-every", "-s", type=int, default=-1,
              help="Increment to save labels and model object (-1 for just last)")(function)
    function = click.option("--save-model", is_flag=True, help="Save model object at the end of training")(function)
    function = click.option("--max-states", "-m", type=int, default=100, help="Maximum number of states")(function)
    function = click.option("--npcs", type=int, default=10, help="Number of PCs to use")(function)
    function = click.option("--whiten", "-w", type=str, default='all', help="Whiten (e)each (a)ll or (n)o whitening")(function)
    function = click.option("--progressbar", "-p", type=bool, default=True, help="Show model progress")(function)
    function = click.option("--percent-split", type=int, default=20, help="Training-validation split percentage")(function)
    function = click.option("--load-groups", "-h", type=bool, default=True, help="Dictates in PC Scores should be loaded with their associated group.")(function)
    function = click.option("--gamma", "-g", type=float, default=1e3, help="Gamma; probability prior distribution for PCs explaining syllable states. Smaller gamma = steeper PC_Scree plot.")(function)
    function = click.option("--alpha", "-a", type=float, default=5.7,
                            help="Alpha; probability prior distribution for syllable transition rate.")(function)
    function = click.option("--noise-level", type=float, default=0, help="Additive white gaussian noise for regularization")(function)
    function = click.option("--nlags", type=int, default=3, help="Number of lags to use")(function)
    function = click.option("--separate-trans", is_flag=True, help="Use separate transition matrix per group")(function)
    function = click.option("--robust", is_flag=True, help="Use tAR model")(function)
    function = click.option("--converge", is_flag=True, help="Use tAR model")(function)
    function = click.option("--tolerance", "-t", type=float, default=1000,
                        help="Tolerance value to check whether model training loglikelihood has converged.")(function)

    return function

def kappa_scan_parameters(function):
    function = click.option('--min-kappa', type=float, default=None,
                            help='Minimum kappa value to train model on.')(function)
    function = click.option('--max-kappa', type=float, default=None,
                            help='Maximum kappa value to train model on.')(function)
    function = click.option('--n-models', type=int, default=10,
                            help='Minimum kappa value to train model on.')(function)
    function = click.option('--prefix', type=str, default='',
                            help='Batch command string to prefix model training command.')(function)
    function = click.option('--cluster-type', type=click.Choice(['local', 'slurm']), default='local',
                            help='Platform to train models on')(function)
    function = click.option('--ncpus', '-n', type=int, default=4, help="Number of CPUs")(function)
    function = click.option('--memory', '-m', type=str, default="5GB", help="RAM string")(function)
    function = click.option('--wall-time', '-w', type=str, default='3:00:00', help="Wall time")(function)
    function = click.option('--partition', type=str, default='short', help="Partition name")(function)

    return function

# this is the entry point for learning models over Kubernetes, expose all
# parameters we could/would possibly scan over
@cli.command(name="learn-model", help="Trains ARHMM on PCA Scores with given training parameters")
@click.argument("input_file", type=click.Path(exists=True))
@click.argument("dest_file", type=click.Path(file_okay=True, writable=True, resolve_path=True))
@modeling_parameters
@click.option("--kappa", "-k", type=float, default=None, help="Kappa; probability prior distribution for syllable duration. Larger k = longer syllable lengths")
@click.option("--checkpoint-freq", type=int, default=-1, help='checkpoint the training after N iterations')
@click.option("--use-checkpoint", is_flag=True, help='indicate whether to use previously saved checkpoint')
@click.option("--index", "-i", type=click.Path(), default="", help="Path to moseq2-index.yaml for group definitions (used only with the separate-trans flag)")
@click.option("--default-group", type=str, default="n/a", help="Default group to use for separate-trans")
@click.option("--verbose", '-v', is_flag=True, help="Print syllable log-likelihoods during training.")
def learn_model(input_file, dest_file, hold_out, hold_out_seed, nfolds, ncpus,
                num_iter, var_name, e_step,
                save_every, save_model, max_states, npcs, whiten, progressbar, percent_split,
                load_groups, kappa, gamma, alpha, noise_level, nlags, separate_trans, robust,
                converge, tolerance, checkpoint_freq, use_checkpoint, index, default_group, verbose):

    click_data = click.get_current_context().params
    learn_model_wrapper(input_file, dest_file, click_data)

@cli.command(name='kappa-scan', help='Batch fit multiple models scanning over different syllable length probability prior.')
@click.argument('input_file', type=click.Path(exists=True))
@click.argument('index_file', type=click.Path(exists=True))
@click.argument('output_dir', type=click.Path(exists=False))
@modeling_parameters
@kappa_scan_parameters
def kappa_scan_fit_models(input_file, index_file, output_dir, hold_out, hold_out_seed, nfolds, ncpus,
                num_iter, var_name, e_step, save_every, save_model, max_states, npcs, whiten, progressbar,
                percent_split, load_groups, gamma, alpha, noise_level, nlags, separate_trans, robust,
                converge, tolerance, min_kappa, max_kappa, n_models, prefix, cluster_type, memory,
                wall_time, partition):

    click_data = click.get_current_context().params
    kappa_scan_fit_models_wrapper(input_file, index_file, output_dir, click_data)


if __name__ == '__main__':
    cli()