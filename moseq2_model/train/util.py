import os
import sys
import numpy as np
from functools import partial
from collections import OrderedDict, defaultdict
from moseq2_model.util import progressbar, save_arhmm_checkpoint, append_resample

def train_model(model, num_iter=100, save_every=1, ncpus=1, checkpoint_freq=None,
                checkpoint_file=None, start=0, save_file=None, progress_kwargs={},
                num_frames=[1], train_data=None, val_data=None, separate_trans=False, groups=None, verbose=False):
    '''

    Parameters
    ----------
    model (ARHMM): model to train.
    num_iter (int): total number of resampling iterations.
    save_every (int): model parameter updating frequency.
    ncpus (int): number of cpus to resample model.
    checkpoint_freq (int): frequency of new checkpoint saves in iterations
    checkpoint_file (str): path to new checkpoint file
    start (int): starting iteration index (used to resume modeling, default is 0).
    save_file (str): path to file to save model checkpoint (only if checkpoint_freq > 0)
    progress_kwargs (dict): keyword arguments for progress bar
    num_frames (int): total number of frames included in modeling
    train_data (OrderedDict): dict of validation data (only if verbose = True)
    val_data (OrderedDict): dict of validation data (only if verbose = True)
    separate_trans (bool): using different transition matrices
    groups (list): list of groups included in modeling (only if verbose = True)
    verbose (bool): Compute model summary.

    Returns
    -------
    model (ARHMM): trained model.
    model.log_likelihood() (list): list of training Log-likelihoods per session after modeling.
    get_labels_from_model(model) (list): list of labels predicted post-modeling.
    iter_lls (list): list of log-likelihoods at an iteration level.
    iter_holls (list): list of held-out log-likelihoods at an iteration level.
    group_idx (list): list of group names per modeled session.
    '''

    checkpoint = checkpoint_freq is not None

    iter_lls = []
    iter_holls = []
    group_idx = ['default']
    for itr in progressbar(range(start, num_iter), **progress_kwargs):
        try:
            model.resample_model(num_procs=ncpus)
        except:
            print("Unexpected error:", sys.exc_info()[0])
            print('Breaking.')
            break
        if verbose:
            iter_lls, iter_holls = get_model_summary(model, groups, train_data, val_data, separate_trans, num_frames, iter_lls, iter_holls)

        # append resample stats to a file
        if (itr + 1) % save_every == 0:
            save_dict = {
                (itr + 1): {
                    'iter': itr + 1,
                    'log_likelihoods': model.log_likelihood(),
                    'labels': get_labels_from_model(model)
                }
            }
            append_resample(save_file, save_dict)
        # checkpoint if needed
        if checkpoint and ((itr + 1) % checkpoint_freq == 0):
            save_data = {
                'iter': itr + 1,
                'model': model,
            }
            save_arhmm_checkpoint(checkpoint_file, save_data)

    if groups != None:
        group_idx = groups
    return model, model.log_likelihood(), get_labels_from_model(model), iter_lls, iter_holls, group_idx


def get_model_summary(model, groups, train_data, val_data, separate_trans, num_frames, iter_lls, iter_holls):
    '''
    Computes a summary of model performance after resampling steps. Is only run if verbose = True.
    Parameters
    ----------
    model (ARHMM): model to compute lls.
    groups (list): list of session group names.
    train_data (OrderedDict): Ordered dict of training data
    val_data: (OrderedDict): Ordered dict of validation/held-out data
    separate_trans (bool) indicates whether to separate lls for each group.
    num_frames (int): total number of frames included in modeling.
    iter_lls (list): list of log-likelihoods at an iteration level.
    iter_holls (list): list of held-out log-likelihoods at an iteration level.

    Returns
    -------
    iter_lls (list): updated list of log-likelihoods at an iteration level.
    iter_holls (list): updated list of held-out log-likelihoods at an iteration level.
    '''

    if not separate_trans:
        train_ll = model.log_likelihood() / sum(num_frames)
        iter_lls.append(train_ll)
    else:
        group_lls = []
        group_idx = []
        if type(groups) == tuple:
            for g in list(set(groups[0])):
                if g != 'n/a':
                    train_ll = [model.log_likelihood(v, group_id=g) for v in train_data.values()]
                    lens = [len(v) for v in train_data.values()]
                    group_lls.append(sum(train_ll) / sum(lens))
                    group_idx.append(g)
        else:
            for g in list(set(groups)):
                if g != 'n/a':
                    train_ll = [model.log_likelihood(v, group_id=g) for v in train_data.values()]
                    lens = [len(v) for v in train_data.values()]
                    group_lls.append(sum(train_ll) / sum(lens))
                    group_idx.append(g)

        iter_lls.append(group_lls)

    if not separate_trans:
        val_ll = [model.log_likelihood(v) for v in val_data.values()]
        lens = [len(v) for v in val_data.values()]
        if len(val_ll) > 1:
            val_ll = sum(val_ll) / sum(lens)
        else:
            try:
                val_ll = sum(val_ll) / len(val_ll[0])
            except:
                val_ll = sum(val_ll) / len(val_ll)
        iter_holls.append(val_ll)
    else:
        group_lls = []
        if type(groups) == tuple:
            for g in list(set(groups[1])):
                if g != 'n/a':
                    val_ll = [model.log_likelihood(v, group_id=g) for v in val_data.values()]
                    lens = [len(v) for v in val_data.values()]
                    group_lls.append(sum(val_ll) / sum(lens))
        else:
            for g in list(set(groups)):
                if g != 'n/a':
                    val_ll = [model.log_likelihood(v, group_id=g) for v in val_data.values()]
                    lens = [len(v) for v in val_data.values()]
                    group_lls.append(sum(val_ll) / sum(lens))
        iter_holls.append(group_lls)

    return iter_lls, iter_holls

def get_labels_from_model(model):
    '''
    Grabs the model labels for each training dataset and places them in a list.
    Parameters
    ----------
    model (ARHMM): trained ARHMM model

    Returns
    -------
    cat_labels (list): Predicted syllable labels for all frames concatenated into a single list.
    '''

    cat_labels = [np.append(np.repeat(-5, model.nlags), s.stateseq) for s in model.states_list]
    return cat_labels


# taken from moseq by @mattjj and @alexbw
def whiten_all(data_dict, center=True):
    '''
    Whitens all the PC Scores at once.
    Parameters
    ----------
    data_dict (OrderedDict): Training dictionary
    center (bool): Indicates whether to center data.

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
    Whiten each group of PC scores separately
    Parameters
    ----------
    data_dict (OrderedDict): Training dictionary
    center (bool): Indicates whether to normalize data.

    Returns
    -------
    data_dict (OrderedDict): Whitened training data dictionary
    '''

    for k, v in data_dict.items():
        tmp_dict = whiten_all(OrderedDict([(k, v)]), center=center)
        data_dict[k] = tmp_dict[k]

    return data_dict
    #return OrderedDict((k, whiten_all(OrderedDict([k,v]), center=center)) for k, v in data_dict.items())

def run_e_step(arhmm):
    '''
    Computes the expected states for each training dataset and places them in a list.
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
    center (bool): Indicates whether to normalize data.

    Returns
    -------
    data_dict (OrderedDict): z-scored training data dictionary
    '''

    for k, v in data_dict.items():
        tmp_dict = zscore_all(OrderedDict([(k, v)]), center=center)
        data_dict[k] = tmp_dict[k]

    return data_dict


def zscore_all(data_dict, npcs=10, center=True):
    '''
    z-score the PC Scores altogether.
    Parameters
    ----------
    data_dict (OrderedDict): Training dictionary
    npcs (int): number of pcs included
    center (bool): Indicates whether to normalize data.

    Returns
    -------
    data_dict (OrderedDict): z-scored training data dictionary
    '''

    valid_scores = np.concatenate([x[~np.isnan(x).any(axis=1), :npcs] for x in data_dict.values()])
    mu, sig = valid_scores.mean(axis=0), valid_scores.std(axis=0)

    for k, v in data_dict.items():
        data_dict[k] = (v - mu) / sig

    return data_dict


# taken from syllables by @alewbw
def get_crosslikes(arhmm, frame_by_frame=False):
    '''
    Compute cross log-likelihood validation ratios.
    Parameters
    ----------
    arhmm (ARHMM): Model to compute cross
    frame_by_frame (bool): Compute cross-lls for each state sequence

    Returns
    -------
    All_CLs (list): cross-log-likelihoods of each state
    CL (np.ndarray): means of all state cross-log-likelihoods
    '''

    all_CLs = defaultdict(list)
    Nstates = arhmm.num_states

    if frame_by_frame:
        for s in arhmm.states_list:
            for j in range(Nstates):
                likes = s.aBl[s.stateseq == j]
                for i in range(Nstates):
                    all_CLs[(i, j)].append(likes[:, i] - likes[:, j])
        all_CLs = defaultdict(
            list,
            {k: np.concatenate(v) for k, v in all_CLs.items()})
    else:
        for s in arhmm.states_list:
            for j in range(Nstates):
                for sl in slices_from_indicators(s.stateseq == j):
                    likes = np.nansum(s.aBl[sl], axis=0)
                    for i in range(Nstates):
                        all_CLs[(i, j)].append(likes[i] - likes[j])

    CL = np.zeros((Nstates, Nstates))
    for (i, j), _ in np.ndenumerate(CL):
        CL[i, j] = np.nanmean(all_CLs[(i, j)])

    return all_CLs, CL


def slices_from_indicators(indseq):
    '''
    Given indices for seqences, return list sliced sublists.
    Parameters
    ----------
    indseq (list): indices to create slices at.

    Returns
    -------
    (list): list of slices from given indices.
    '''

    return [sl for sl in rleslices(indseq) if indseq[sl.start]]


def rleslices(seq):
    '''
    Get changepoint index slices
    Parameters
    ----------
    seq (list): list of labels

    Returns
    -------
    (map generator): slices given syllable changepoint indices
    '''

    pos, = np.where(np.diff(seq) != 0)
    pos = np.concatenate(([0], pos+1, [len(seq)]))
    return map(slice, pos[:-1], pos[1:])
