import numpy as np
from autoregressive.distributions import AutoRegression
from pybasicbayes.distributions import RobustAutoRegression
from autoregressive.models import ARWeakLimitStickyHDPHMM, ARWeakLimitStickyHDPHMMSeparateTrans, \
    FastARWeakLimitStickyHDPHMM, FastARWeakLimitStickyHDPHMMSeparateTrans
from moseq2_model.util import merge_dicts
import warnings


# Empirical bayes estimate of S_0 (from MoSeq)
def _get_empirical_ar_params(train_datas, params):
    """
    Estimate the parameters of an AR observation model
    by fitting a single AR model to the entire dataset.
    """

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

    # make sure to include nu if we're using the tAR model
    if 'nu' in params.keys():
        obs_params['nu'] = params['nu']

    return obs_params


def ARHMM(data_dict, kappa=1e6, gamma=999, nlags=3, nu=4,
          K_0_scale=10.0, S_0_scale=0.01, max_states=100, empirical_bayes=True,
          affine=True, model_hypparams={}, obs_hypparams={}, sticky_init=False,
          separate_trans=False, groups=None, robust=False):

    warnings.filterwarnings("ignore", category=np.VisibleDeprecationWarning)

    data_dim = list(data_dict.values())[0].shape[1]

    default_obs_hypparams = {
        'nu_0': data_dim+2,
        'S_0': S_0_scale*np.eye(data_dim),
        'M_0': np.hstack((np.eye(data_dim),
                          np.zeros((data_dim, data_dim * (nlags-1))),
                          np.zeros((data_dim, int(affine))))),
        'affine': affine,
        'K_0': K_0_scale*np.eye(data_dim*nlags+affine)
        }

    default_model_hypparams = {
        'alpha': 5.7,
        'gamma': gamma,
        'kappa': kappa,
        'init_state_distn': 'uniform'
        }

    if robust:
        default_obs_hypparams['nu'] = nu

    obs_hypparams = merge_dicts(default_obs_hypparams, obs_hypparams)
    model_hypparams = merge_dicts(default_model_hypparams, model_hypparams)

    if empirical_bayes:
        obs_hypparams = _get_empirical_ar_params(list(data_dict.values()), obs_hypparams)

    # TODO: return initialization parameters for saving downstream

    if separate_trans and not robust:
        print('Using model class FastARWeakLimitStickyHDPHMMSeparateTrans')
        obs_distns = [AutoRegression(**obs_hypparams) for _ in range(max_states)]
        model = FastARWeakLimitStickyHDPHMMSeparateTrans(obs_distns=obs_distns, **model_hypparams)
    elif not separate_trans and not robust:
        print('Using model class FastARWeakLimitStickyHDPHMM')
        obs_distns = [AutoRegression(**obs_hypparams) for _ in range(max_states)]
        model = FastARWeakLimitStickyHDPHMM(obs_distns=obs_distns, **model_hypparams)
    elif not separate_trans and robust:
        print('Using ROBUST model class ARWeakLimitStickyHDPHMM')
        obs_distns = [RobustAutoRegression(**obs_hypparams) for _ in range(max_states)]
        model = ARWeakLimitStickyHDPHMM(obs_distns=obs_distns, **model_hypparams)
    elif separate_trans and robust:
        print('Using ROBUST model class ARWeakLimitStickyHDPHMMSeparateTrans')
        obs_distns = [RobustAutoRegression(**obs_hypparams) for _ in range(max_states)]
        model = ARWeakLimitStickyHDPHMMSeparateTrans(obs_distns=obs_distns, **model_hypparams)

    # add ze data

    for index, (data_name, data) in enumerate(data_dict.items()):
        print('Adding data from key {}'.format(str(data_name)))
        if separate_trans:
            print('Group ID: {}'.format(str(groups[index])))
            model.add_data(data, group_id=groups[index])
        else:
            model.add_data(data)

    # initialize ze states per SL's recommendation

    if sticky_init:
        for i in range(0, len(model.stateseqs)):
            seqlen = len(model.stateseqs[i])
            z_init = np.random.randint(max_states, size=seqlen//10).repeat(10)
            z_init = np.append(z_init, np.random.randint(max_states, size=seqlen-len(z_init)))
            model.stateseqs[i] = z_init.copy().astype('int32')

    return model
