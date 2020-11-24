'''
GUI front-end function for training ARHMM.
'''

import ruamel.yaml as yaml
from .cli import learn_model
from os.path import dirname, join, exists
from moseq2_model.helpers.wrappers import learn_model_wrapper, kappa_scan_fit_models_wrapper

def learn_model_command(progress_paths, hold_out=False, nfolds=2, num_iter=100,
                        max_states=100, npcs=10, scan_scale='log', kappa=None, min_kappa=None, max_kappa=None, n_models=5,
                        alpha=5.7, gamma=1e3, separate_trans=True, robust=True, checkpoint_freq=-1, use_checkpoint=False,
                        check_every=5, select_groups=False, percent_split=0, output_dir=None,
                        out_script='train_out.sh', cluster_type='local', get_cmd=True, run_cmd=False, prefix='',
                        memory='16GB', wall_time='3:00:00', partition='short', verbose=False):
    '''
    Trains ARHMM from Jupyter notebook. Note that the configuration file parameters will be overriden with the
    inputted parameters from the jupyter notebook cell function call.

    Parameters
    ----------
    progress_paths (dict): notebook progress dict that contains paths to the pca scores, config, and index files.
    hold_out (bool): indicate whether to hold out data or use train_test_split.
    nfolds (int): number of folds to hold out.
    num_iter (int): number of training iterations.
    max_states (int): maximum number of model states.
    npcs (int): number of PCs to include in analysis.
    kappa (float): probability prior distribution for syllable duration. Larger kappa = longer syllable durations.
    min_kappa (float): Minimum kappa to train model on. 
    max_kappa (float): Maximum kappa to train model on.
    n_models (int): Number of models to spawn to scan kappa values
    scan_scale (str): Scale factor to generate scanning kappa values. ['log', 'linear']
    separate_trans (bool): indicate whether to compute separate syllable transition matrices for each group.
    robust (bool): indicate whether to use a t-distributed syllable label distribution. (robust-ARHMM)
    checkpoint_freq (int): frequency at which to save model checkpoints
    use_checkpoint (bool): indicates to load a previously saved checkpoint
    alpha (float): probability prior distribution for syllable transition rate.
    gamma (float): probability prior distribution for PCs explaining syllable states. Smaller gamma = steeper PC_Scree plot.
    select_groups (bool): indicates to display all sessions and choose subset of groups to model alone.
    check_every (int): number of iterations between each training iteration log-likelihood check.
    select_groups (bool): indicates whether to interactively select data to model by group name.
    get_cmd (bool): indicates to print all the kappa scan learn-model command outputs.
    run_cmd (bool): indicates to run all the kappa scan learn-model commands.
    percent_split (int): train-validation data split ratio percentage.
    output_dir (str): directory to store multiple trained models via kappa-scan
    out_script (str): name of the script containing all the kappa scanning commands.
    cluster_type (str): name of cluster to run model training on; either ['local', 'slurm']
    prefix (str): slurm command prefix with job specification parameters.
    memory (str): amount of memory in GB to allocate to each training job.
    wall_time (str): maximum time for a slurm job to run.
    partition (str): slurm partition name to run training jobs on.
    verbose (bool): compute modeling summary (Warning current implementation is can slow down training).

    Returns
    -------
    None
    '''

    # Load proper input variables
    input_file = progress_paths['scores_path']
    dest_file = progress_paths['model_path']
    config_file = progress_paths['config_file']
    index = progress_paths['index_file']

    assert exists(input_file), "PCA Scores not found; set the correct path in progress_paths['scores_path']"
    assert exists(config_file), "Config file not found; set the correct path in progress_paths['config_file']"
    assert exists(index), "Index file not found; set the correct path in progress_paths['index_file']"

    with open(config_file, 'r') as f:
        config_data = yaml.safe_load(f)

    # Get default CLI params
    params = {tmp.name: tmp.default for tmp in learn_model.params if not tmp.required}
    # merge default params and config data, preferring values in config data
    config_data = {**params, **config_data}

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
    config_data['check_every'] = check_every
    config_data['index'] = index

    if output_dir is None:
        print('Output directory not specified, saving models to base directory.')
        output_dir = dirname(index)

    config_data['out_script'] = join(output_dir, out_script)

    if kappa == 'scan':
        assert any(scan_scale == x for x in ('log', 'linear')), 'scan scale must be "log" or "linear"'
        config_data['scan_scale'] = scan_scale
        config_data['min_kappa'] = min_kappa
        config_data['max_kappa'] = max_kappa
        config_data['n_models'] = n_models

        config_data['cluster_type'] = cluster_type
        config_data['prefix'] = prefix
        config_data['memory'] = memory
        config_data['partition'] = partition
        config_data['wall_time'] = wall_time
        config_data['get_cmd'] = get_cmd
        config_data['run_cmd'] = run_cmd

        command = kappa_scan_fit_models_wrapper(input_file, config_data, output_dir)
        return command
    else:
        learn_model_wrapper(input_file, dest_file, config_data)