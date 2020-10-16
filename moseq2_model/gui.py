'''
GUI front-end function for training ARHMM.
'''

import os
import ruamel.yaml as yaml
from .cli import learn_model
from moseq2_model.helpers.wrappers import learn_model_wrapper, kappa_scan_fit_models_wrapper

def learn_model_command(progress_paths, hold_out=False, nfolds=2, num_iter=100,
                        max_states=100, npcs=10, kappa=None, min_kappa=None, max_kappa=None, n_models=5, alpha=5.7,
                        gamma=1e3, separate_trans=True, robust=True, checkpoint_freq=-1, use_checkpoint=False,
                        select_groups=False, percent_split=20, output_dir=None, cluster_type='local', get_cmd=True,
                        verbose=False):
    '''
    Trains ARHMM from Jupyter notebook.

    Parameters
    ----------
    progress_paths (dict):
    hold_out (bool): indicate whether to hold out data or use train_test_split.
    nfolds (int): number of folds to hold out.
    num_iter (int): number of training iterations.
    max_states (int): maximum number of model states.
    npcs (int): number of PCs to include in analysis.
    kappa (float): probability prior distribution for syllable duration. Larger kappa = longer syllable durations.
    separate_trans (bool): indicate whether to compute separate syllable transition matrices for each group.
    robust (bool): indicate whether to use a t-distributed syllable label distribution. (robust-ARHMM)
    checkpoint_freq (int): frequency at which to save model checkpoints
    use_checkpoint (bool): indicates to load a previously saved checkpoint
    alpha (float): probability prior distribution for syllable transition rate.
    gamma (float): probability prior distribution for PCs explaining syllable states. Smaller gamma = steeper PC_Scree plot.
    select_groups (bool): indicates to display all sessions and choose subset of groups to model alone.
    percent_split (int): train-validation data split ratio percentage.
    verbose (bool): compute modeling summary (Warning current implementation is slow).

    Returns
    -------
    None
    '''

    # Load proper input variables
    input_file = progress_paths['scores_path']
    dest_file = progress_paths['model_path']
    config_file = progress_paths['config_file']
    index = progress_paths['index_file']

    with open(config_file, 'r') as f:
        config_data = yaml.safe_load(f)

    # Get default CLI params
    params = {tmp.name: tmp.default for tmp in learn_model.params if not tmp.required}
    # merge default params and config data, preferring values in config data
    config_data = {**params, **config_data}

    # TODO: does the documentation reflect that the parameters in the learn_model_command function
    # will override the ones set in the config file?
    config_data['alpha'] = alpha
    config_data['gamma'] = gamma
    config_data['kappa'] = kappa
    config_data['separate_trans'] = separate_trans
    config_data['robust'] = robust
    config_data['checkpoint_freq'] = checkpoint_freq
    config_data['hold_out'] = hold_out
    config_data['nfolds'] = nfolds
    config_data['num_iter'] = num_iter
    config_data['max_states'] = max_states
    config_data['npcs'] = npcs
    config_data['percent_split'] = percent_split
    config_data['verbose'] = verbose
    config_data['select_groups'] = select_groups
    config_data['use_checkpoint'] = use_checkpoint

    # TODO: kappa scan should be a separate gui function, each takes different parameters
    # TODO: none of the slurm keywords that are found in the cli are found here in the gui version of scan
    if kappa == 'scan':
        config_data['min_kappa'] = min_kappa
        config_data['max_kappa'] = max_kappa
        config_data['n_models'] = n_models
        config_data['get_cmd'] = get_cmd
        config_data['cluster_type'] = cluster_type

        if output_dir == None:
            print('Output directory not specified, saving models to base directory.')
            output_dir = os.path.dirname(index)

        command = kappa_scan_fit_models_wrapper(input_file, index, config_data, output_dir)
        return command
    else:
        learn_model_wrapper(input_file, dest_file, config_data, index)