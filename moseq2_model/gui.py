import ruamel.yaml as yaml
from moseq2_model.helpers.wrappers import learn_model_wrapper

def learn_model_command(input_file, dest_file, config_file, index, hold_out, nfolds, num_iter,
                max_states, npcs, kappa, separate_trans, robust, checkpoint_freq,
                        percent_split=20, verbose=False, output_directory=None):

    alpha = 5.7
    gamma = 1e3

    with open(config_file, 'r') as f:
        config_data = yaml.safe_load(f)

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

    learn_model_wrapper(input_file, dest_file, config_data, index, output_directory=output_directory, gui=True)
