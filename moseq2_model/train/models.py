'''
ARHMM model initialization utilities.
'''

import warnings
import numpy as np
from cytoolz import merge
from functools import partial
from autoregressive.distributions import AutoRegression
from pybasicbayes.distributions import RobustAutoRegression
from autoregressive.models import ARWeakLimitStickyHDPHMM, ARWeakLimitStickyHDPHMMSeparateTrans, \
    FastARWeakLimitStickyHDPHMM, FastARWeakLimitStickyHDPHMMSeparateTrans


flush_print = partial(print, flush=True)

# Empirical bayes estimate of S_0 (from MoSeq)
def _get_empirical_ar_params(train_datas, params):
    '''
    Estimate the parameters of an AR observation model
    by fitting a single AR model to the entire dataset.

    Parameters
    ----------
    train_datas (list): list of np.ndarrays representing each session's PC scores
    params (dict): dict containing modeling parameters used for initialization

    Returns
    -------
    obs_params (dict): dict of observed parameters to use in modeling.
    '''

    assert isinstance(train_datas, list) and len(train_datas) > 0
    datadimension = train_datas[0].shape[1]
    assert params["nu_0"] > datadimension + 1

    # Initialize the observation parameters
    obs_params = dict(nu_0=params["nu_0"],
                      S_0=params['S_0'],
                      M_0=params['M_0'],
                      K_0=params['K_0'],
                      affine=params['affine'])

    # Fit an AR model to the entire dataset
    obs_distn = AutoRegression(**obs_params)
    obs_distn.max_likelihood(train_datas)

    # Use the inferred noise covariance as the prior mean
    # E_{IW}[S] = S_0 / (nu_0 - datadimension - 1)
    obs_params["S_0"] = obs_distn.sigma * (params["nu_0"] - datadimension - 1)

    return obs_params


def ARHMM(data_dict, kappa=1e6, gamma=999, nlags=3, alpha=5.7,
          K_0_scale=10.0, S_0_scale=0.01, max_states=100, empirical_bayes=True,
          affine=True, model_hypparams={}, obs_hypparams={}, sticky_init=False,
          separate_trans=False, groups=None, robust=False, silent=False):
    '''
    Initializes ARHMM and adds data and group labels to the ARHMM model.

    Parameters
    ----------
    data_dict (OrderedDict): training data to add to model
    kappa (float): hyperparameter for setting syllable duration. Larger kappa = longer syllable durations
    gamma (float): scaling parameter for hierarchical dirichlet process (it's recommended to leave this parameter alone)
    nlags (int): number of lag frames to add to sessions
    alpha (float): scaling parameter for hierarchical dirichlet process (it's recommended to leave this parameter alone)
    K_0_scale (float): Standard deviation of lagged data
    S_0_scale (float): scale standard deviation initialization (don't touch this parameter unless necessary)
    max_states (int): Maximum number of model states
    empirical_bayes (bool): Use empirical bayes to initialize sigma
    affine (bool): Use affine transformation in the AR processes
    model_hypparams (dict): other model parameters (don't touch this parameter unless necessary)
    obs_hypparams (dict): observed parameters nu_0, S_0, M_0, and K_0 (don't touch this parameter unless necessary)
    sticky_init (bool): Initialize the model with random states
    separate_trans (bool): use separate transition matrices for each group
    groups (list): list of groups to model
    robust (bool): use student's t-distributed AR model
    silent (bool): flag to print out model information.

    Returns
    -------
    model (ARHMM): initialized model object
    '''

    warnings.filterwarnings("ignore", category=np.VisibleDeprecationWarning)

    # Get dimension of training data
    data_dim = list(data_dict.values())[0].shape[1]

    # Set observational hyper-parameters
    default_obs_hypparams = {
        'nu_0': data_dim + 2,
        'S_0': S_0_scale * np.eye(data_dim),
        'M_0': np.hstack((np.eye(data_dim),
                          np.zeros((data_dim, data_dim * (nlags - 1))),
                          np.zeros((data_dim, int(affine))))),
        'affine': affine,
        'K_0': K_0_scale * np.eye(data_dim * nlags + affine)
        }

    # Set modeling hyper-parameters
    default_model_hypparams = {
        'alpha': alpha,
        'gamma': gamma,
        'kappa': kappa,
        'init_state_distn': 'uniform'
        }

    # Update hyper-parameters given function inputted dict objects
    obs_hypparams = merge(default_obs_hypparams, obs_hypparams)
    model_hypparams = merge(default_model_hypparams, model_hypparams)

    # Use empirical_bayes to set initial sigma values
    if empirical_bayes:
        obs_hypparams = _get_empirical_ar_params(list(data_dict.values()), obs_hypparams)

    if separate_trans and not robust:
        # Loading C-accelerated model with separate transition graphs
        if not silent:
            flush_print('Using model class FastARWeakLimitStickyHDPHMMSeparateTrans')
        obs_distns = [AutoRegression(**obs_hypparams) for _ in range(max_states)]
        model = FastARWeakLimitStickyHDPHMMSeparateTrans(obs_distns=obs_distns, **model_hypparams)
    elif not separate_trans and not robust:
        # Loading default C-accelerated model
        if not silent:
            flush_print('Using model class FastARWeakLimitStickyHDPHMM')
        obs_distns = [AutoRegression(**obs_hypparams) for _ in range(max_states)]
        model = FastARWeakLimitStickyHDPHMM(obs_distns=obs_distns, **model_hypparams)
    elif not separate_trans and robust:
        # Loading t-distributed ARHMM
        if not silent:
            flush_print('Using ROBUST model class ARWeakLimitStickyHDPHMM')
        obs_distns = [RobustAutoRegression(**obs_hypparams) for _ in range(max_states)]
        model = ARWeakLimitStickyHDPHMM(obs_distns=obs_distns, **model_hypparams)
    elif separate_trans and robust:
        # Loading t-distributed ARHMM with separate transition graphs
        if not silent:
            flush_print('Using ROBUST model class ARWeakLimitStickyHDPHMMSeparateTrans')
        obs_distns = [RobustAutoRegression(**obs_hypparams) for _ in range(max_states)]
        model = ARWeakLimitStickyHDPHMMSeparateTrans(obs_distns=obs_distns, **model_hypparams)

    print(f'Total number of included sessions: {len(data_dict)}')
    for data_name, data in data_dict.items():
        # Add data to model
        if not silent:
            flush_print(f'Adding data from key {data_name}')
        if separate_trans:
            # Optionally add data with corresponding group for separate transition graphs
            if groups[data_name] != 'n/a':
                if not silent:
                    flush_print(f'Group ID: {groups[data_name]}')
                model.add_data(data, group_id=groups[data_name])
        else:
            # Load data without group, yielding single transition graph
            model.add_data(data)

    # initialize states per SL's recommendation
    if sticky_init:
        for i in range(0, len(model.stateseqs)):
            seqlen = len(model.stateseqs[i])
            z_init = np.random.randint(max_states, size=seqlen // 10).repeat(10)
            z_init = np.append(z_init, np.random.randint(max_states, size=seqlen - len(z_init)))
            model.stateseqs[i] = z_init.copy().astype('int32')

    return model