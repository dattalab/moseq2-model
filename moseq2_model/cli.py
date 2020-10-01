'''
CLI front-end operations. This module contains all the functionality and configurable parameters
users can alter to most accurately process their data.
'''

import click
import numpy as np
from moseq2_model.util import load_pcs
from moseq2_model.train.models import flush_print
from moseq2_model.helpers.wrappers import learn_model_wrapper

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


# this is the entry point for learning models over Kubernetes, expose all
# parameters we could/would possibly scan over
@cli.command(name="learn-model", help="Trains ARHMM on PCA Scores with given training parameters")
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
@click.option("--percent-split", type=int, default=20, help="Training-validation split percentage")
@click.option("--load-groups", "-h", type=bool, default=True, help="Dictates in PC Scores should be loaded with their associated group.")
@click.option("--kappa", "-k", type=float, default=None, help="Kappa; probability prior distribution for syllable duration. Larger k = longer syllable lengths")
@click.option("--gamma", "-g", type=float, default=1e3, help="Gamma; probability prior distribution for PCs explaining syllable states. Smaller gamma = steeper PC_Scree plot.")
@click.option("--alpha", "-a", type=float, default=5.7, help="Alpha; probability prior distribution for syllable transition rate.")
@click.option("--noise-level", type=float, default=0, help="Additive white gaussian noise for regularization")
@click.option("--nlags", type=int, default=3, help="Number of lags to use")
@click.option("--separate-trans", is_flag=True, help="Use separate transition matrix per group")
@click.option("--robust", is_flag=True, help="Use tAR model")
@click.option("--converge", is_flag=True, help="Stop model training when log-likelihood value converges.")
@click.option("--tolerance", "-t", type=float, default=1000, help="Tolerance value to check whether model training loglikelihood has converged.")
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


if __name__ == '__main__':
    cli()