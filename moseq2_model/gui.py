import ruamel.yaml as yaml
from .cli import learn_model
from moseq2_model.helpers.wrappers import learn_model_wrapper

def learn_model_command(input_file, dest_file, config_file, index, hold_out, nfolds, num_iter,
                max_states, npcs, kappa, separate_trans, robust, checkpoint_freq, use_checkpoint=False,
                        alpha=5.7, gamma=1e3, select_groups=False, percent_split=20, verbose=False, output_directory=None):
    '''
    Trains ARHMM from Jupyter notebook.

    Parameters
    ----------
    input_file (str): pca scores file path.
    dest_file (str): path to save model to.
    config_file (str): configuration file path.
    index (str): index file path.
    hold_out (bool): indicate whether to hold out data or use train_test_split.
    nfolds (int): number of folds to hold out.
    num_iter (int): number of training iterations.
    max_states (int): maximum number of model states.
    npcs (int): number of PCs to include in analysis.
    kappa (float): probability prior distribution for syllable duration.
    separate_trans (bool): indicate whether to compute separate syllable transition matrices for each group.
    robust (bool): indicate whether to use a t-distributed syllable label distribution. (robust-ARHMM)
    checkpoint_freq (int): frequency at which to save model checkpoints
    percent_split (int): train-validation data split ratio percentage.
    verbose (bool): compute modeling summary (Warning current implementation is slow).
    output_directory (str): alternative output directory for GUI users

    Returns
    -------
    None
    '''

    with open(config_file, 'r') as f:
        config_data = yaml.safe_load(f)

    config_data['alpha'] = config_data.get('alpha', 5.7)
    config_data['gamma'] = config_data.get('gamma', 1e3)
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

    # Get default CLI params
    objs = learn_model.params

    params = {tmp.name: tmp.default for tmp in objs if not tmp.required}
    for k, v in params.items():
        if k not in config_data.keys():
            config_data[k] = v

    learn_model_wrapper(input_file, dest_file, config_data, index, output_directory=output_directory, gui=True)