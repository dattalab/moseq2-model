'''
ARHMM utility functions
'''

import math
import numpy as np
from cytoolz import valmap
from tqdm.auto import tqdm
from scipy.stats import norm
from functools import partial
from collections import OrderedDict, defaultdict
from moseq2_model.util import save_arhmm_checkpoint, get_loglikelihoods


def train_model(model, num_iter=100, ncpus=1, checkpoint_freq=None,
                checkpoint_file=None, start=0, progress_kwargs={},
                train_data=None, val_data=None, separate_trans=False, groups=None, 
                verbose=False, check_every=2):
    '''
    ARHMM training: Resamples ARHMM for inputted number of iterations,
    and optionally computes loglikelihood scores for each iteration if verbose is True.

    Parameters
    ----------
    model (ARHMM): model to train
    num_iter (int): total number of resampling iterations
    ncpus (int): number of cpus used to resample the model
    checkpoint_freq (int): frequency (iterations) to save a checkpoint of the model
    checkpoint_file (str): path to save new checkpoint file
    start (int): starting iteration index used to resume modeling. Default is 0
    progress_kwargs (dict): keyword arguments for progress bar
    train_data (OrderedDict): dict of training data used for getting log-likelihods
        if verbose is True
    val_data (OrderedDict): dict of validation data used for getting validation
        log-likelihoods if verbose is True.
    separate_trans (bool): use separated transition matrices for each group
    groups (list): list of groups included in modeling used for getting log-likelihoods
        if verbose is True
    verbose (bool): get log-likelihoods at `check_every` interval
    check_every (int): frequency (iterations) to record model training/validation
        log-likelihoods during training

    Returns
    -------
    model (ARHMM): trained model.
    log_likelihood (list): list of training log-likelihoods per session after modeling.
    labels (list): list of labels predicted per session after modeling.
    iter_lls (list): list of training log-likelihoods for each `check_every` iteration.
    iter_holls (list): list of held-out log-likelihoods for each `check_every` iteration.
    interrupt (bool): flag to notify the caller of this function if a keyboard interrupt happened
    '''

    # Checkpointing boolean
    checkpoint = checkpoint_freq is not None

    iter_lls, iter_holls = [], []

    for itr in tqdm(range(start, num_iter), **progress_kwargs):
        # Resample states, and gracefully return in case of a keyboard interrupt
        try:
            model.resample_model(num_procs=ncpus)
        except KeyboardInterrupt:
            print('Training manually interrupted.')
            print('Returning and saving current iteration of model. ')
            return model, model.log_likelihood(), get_labels_from_model(model), iter_lls, iter_holls, True

        summ_stats = {
            'model': model,
            'groups': groups,
            'train_data': train_data,
            'val_data': val_data,
            'separate_trans': separate_trans
        }

        if verbose and ((itr + 1) % check_every == 0):
            # Compute and save iteration training and validation log-likelihoods
            train_ll, ho_ll = get_model_summary(**summ_stats)
            iter_lls.append(train_ll)
            if ho_ll is not None:
                iter_holls.append(ho_ll)

        # checkpoint if needed
        if checkpoint and ((itr + 1) % checkpoint_freq == 0):
            training_checkpoint(model, itr, checkpoint_file)

    return model, model.log_likelihood(), get_labels_from_model(model), iter_lls, iter_holls, False


def training_checkpoint(model, itr, checkpoint_file):
    '''
    Formats the model checkpoint filename and saves the model checkpoint

    Parameters
    ----------
    model (ARHMM): Model being trained.
    itr (itr): Current modeling iteration.
    checkpoint_file (str): Model checkpoint file name.

    Returns
    -------
    '''

    # Pack the data to save in checkpoint
    save_data = {
        'iter': itr + 1,
        'model': model,
        'log_likelihoods': model.log_likelihood(),
        'labels': get_labels_from_model(model)
    }

    # Format checkpoint filename
    checkpoint_file = f'{checkpoint_file}-checkpoint_{itr}.arhmm'

    # Save checkpoint
    save_arhmm_checkpoint(checkpoint_file, save_data)


def get_model_summary(model, groups, train_data, val_data, separate_trans):
    '''
    Computes log-likelihood of train_data and val_data (if not None). aIs only run if verbose = True.

    Parameters
    ----------
    model (ARHMM): model to compute lls.
    groups (list): list of session group names.
    train_data (OrderedDict): Ordered dict of training data
    val_data: (OrderedDict or None): Ordered dict of validation/held-out data
    separate_trans (bool) indicates whether to separate lls for each group.

    Returns
    -------
    train_ll (float): normalized average training log-likelihood across all recording sessions.
    val_ll (float): normalized average held-out log-likelihood across all recording sessions.
    '''
    # Get train and validation groups
    if groups is not None:
        train_groups, val_groups = groups
    else:
        # if there are no groups, separate trans cannot be True
        separate_trans = False
        train_groups, val_groups = None, None

    # Compute normalized log-likelihoods for each session
    train_ll = get_loglikelihoods(model, train_data, train_groups,
                                  separate_trans, normalize=True)

    # return early if there is no validation data
    if val_data is None:
        return np.mean(train_ll), None

    # Get iteration heldout/validation log-likelihood values
    val_ll = get_loglikelihoods(model, val_data, val_groups,
                                separate_trans, normalize=True)

    return np.mean(train_ll), np.mean(val_ll)


def get_labels_from_model(model):
    '''
    Grabs model labels for each training dataset and places them in a list.

    Parameters
    ----------
    model (ARHMM): trained ARHMM model

    Returns
    -------
    labels (list): An array of predicted syllable labels for each training session
    '''

    labels = [np.append(np.repeat(-5, model.nlags), s.stateseq) for s in model.states_list]
    return labels


# taken from moseq by @mattjj and @alexbw
def whiten_all(data_dict, center=True):
    '''
    Whitens the PC Scores (with Cholesky decomposition) using all
    the data to compute the covariance matrix.

    Parameters
    ----------
    data_dict (OrderedDict): Training dataset
    center (bool): Indicates whether to center data by subtracting the mean PC score.

    Returns
    -------
    data_dict (OrderedDict): Whitened training data dictionary
    '''

    non_nan = lambda x: x[~np.isnan(np.reshape(x, (x.shape[0], -1))).any(1)]
    meancov = lambda x: (x.mean(0), np.cov(x, rowvar=False, bias=1))
    contig = partial(np.require, dtype=np.float64, requirements='C')

    mu, Sigma = meancov(np.concatenate(list(map(non_nan, data_dict.values()))))
    L = np.linalg.cholesky(Sigma)

    offset = 0. if center else mu
    apply_whitening = lambda x:  np.linalg.solve(L, (x-mu).T).T + offset

    return OrderedDict((k, contig(apply_whitening(v))) for k, v in data_dict.items())


# taken from moseq by @mattjj and @alexbw
def whiten_each(data_dict, center=True):
    '''
    Whiten the PC scores for each training dataset separately.

    Parameters
    ----------
    data_dict (OrderedDict): Training dataset
    center (bool): Indicates whether to center data by subtracting the mean PC score.

    Returns
    -------
    data_dict (OrderedDict): Whitened training data dictionary
    '''

    for k, v in data_dict.items():
        tmp_dict = whiten_all({k: v}, center=center)
        data_dict[k] = tmp_dict[k]

    return data_dict


def run_e_step(arhmm):
    '''
    Computes the expectation for each state across all frames of the training dataset
    and places them in a list.

    Parameters
    ----------
    arhmm (ARHMM): model to compute expected states from.

    Returns
    -------
    e_states (list): list of expected states
    '''

    arhmm._E_step()
    return [s.expected_states for s in arhmm.states_list]


def zscore_each(data_dict, center=True):
    '''
    z-score each set of PC Scores separately

    Parameters
    ----------
    data_dict (OrderedDict): Training dictionary
    center (bool): Indicates whether to center data by subtracting the mean PC score.

    Returns
    -------
    data_dict (OrderedDict): z-scored training data dictionary
    '''

    for k, v in data_dict.items():
        tmp_dict = zscore_all({k: v}, center=center)
        data_dict[k] = tmp_dict[k]

    return data_dict


def zscore_all(data_dict, npcs=10, center=True):
    '''
    z-score the PC Scores altogether.

    Parameters
    ----------
    data_dict (OrderedDict): Training dictionary
    npcs (int): number of pcs included
    center (bool): Indicates whether to center data by subtracting the mean PC score.

    Returns
    -------
    data_dict (OrderedDict): z-scored training data dictionary
    '''

    scores = np.concatenate(list(data_dict.values()))[:, :npcs]
    mu, sig = np.nanmean(scores, axis=0), np.nanstd(scores, axis=0)

    for k, v in data_dict.items():
        numerator = v[:, :npcs] - mu if center else v[:, :npcs]
        data_dict[k] = numerator / sig

    return data_dict


# taken from syllables by @alexbw
def get_crosslikes(arhmm, frame_by_frame=False):
    '''
    Gets the cross-likelihoods, a measure of confidence in label
    segmentation, for each model label.

    Parameters
    ----------
    arhmm: the ARHMM model object
    frame_by_frame (bool): if True, the cross-likelihoods will be computed for each frame.

    Returns
    -------
    All_CLs (list): a dictionary containing cross-likelihoods for each syllable pair.
    if ``frame_by_frame=True``, it will contain a value for each frame
    CL (np.ndarray): the average cross-likelihood for each syllable pair
    '''

    all_CLs = defaultdict(list)
    Nstates = arhmm.num_states

    if frame_by_frame:
        # Optionally compute Cross-state log-likelihoods over all frame
        for s in arhmm.states_list:
            for j in range(Nstates):
                likes = s.aBl[s.stateseq == j]
                for i in range(Nstates):
                    all_CLs[(i, j)].append(likes[:, i] - likes[:, j])
        all_CLs = valmap(np.concatenate, all_CLs)
    else:
        # Compute cross log-likelihoods across all modeling states
        for s in arhmm.states_list:
            for j in range(Nstates):
                for sl in slices_from_indicators(s.stateseq == j):
                    likes = np.nansum(s.aBl[sl], axis=0)
                    for i in range(Nstates):
                        all_CLs[(i, j)].append(likes[i] - likes[j])

    # Pack cross log-likelihoods into a square confusion matrix
    CL = np.zeros((Nstates, Nstates))
    for (i, j), _ in np.ndenumerate(CL):
        CL[i, j] = np.nanmean(all_CLs[(i, j)])

    return all_CLs, CL


def slices_from_indicators(indseq):
    '''
    Compute start and stop indices (slices) for each contiguous sequence of True values in
    `indseq`.

    Parameters
    ----------
    indseq (list): Indicator array, containing True and False values

    Returns
    -------
    (list): list of slices from `indseq`.
    '''

    return [sl for sl in rleslices(indseq) if indseq[sl.start]]


def rleslices(seq):
    '''
    Get changepoint slices

    Parameters
    ----------
    seq (list): list of labels

    Returns
    -------
    (map generator): slices of syllable changepoints
    '''

    pos, = np.where(np.diff(seq) != 0)
    pos = np.concatenate(([0], pos+1, [len(seq)]))
    return map(slice, pos[:-1], pos[1:])