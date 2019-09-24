import numpy as np
from functools import partial
from collections import OrderedDict, defaultdict
from moseq2_model.util import progressbar


# based on moseq by @mattjj and @alexbw
def train_model(model, num_iter=100, save_every=1, ncpus=1, cli=False, **kwargs):

    # per conversations w/ @mattjj, the fast class of models use openmp no need
    # for "extra" parallelism

    log_likelihoods = []
    labels = []

    for itr in progressbar(range(num_iter), cli=cli, **kwargs):
        model.resample_model(num_procs=ncpus)
        if (np.mod(itr+1, save_every) == 0 or
                np.mod(itr+1, num_iter) == 0):
            log_likelihoods.append(model.log_likelihood())
            seq_list = [s.stateseq for s in model.states_list]
            for seq_itr in range(len(seq_list)):
                seq_list[seq_itr] = np.append(np.repeat(-5, model.nlags), seq_list[seq_itr])
            labels.append(seq_list)

    labels_cat = []

    for i in range(len(labels[0])):
        labels_cat.append(np.array([tmp[i] for tmp in labels], dtype=np.int16))

    return model, log_likelihoods, labels_cat


# simple function for grabbing model labels across the dict
def get_labels_from_model(model):
    cat_labels = [s.stateseq for s in model.states_list]
    return cat_labels


# taken from moseq by @mattjj and @alexbw
def whiten_all(data_dict, center=True):
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
    for k, v in data_dict.items():
        tmp_dict = whiten_all(OrderedDict([(k, v)]), center=center)
        data_dict[k] = tmp_dict[k]

    return data_dict
    #return OrderedDict((k, whiten_all(OrderedDict([k,v]), center=center)) for k, v in data_dict.items())


def zscore_each(data_dict, center=True):
    for k, v in data_dict.items():
        tmp_dict = zscore_all(OrderedDict([(k, v)]), center=center)
        data_dict[k] = tmp_dict[k]

    return data_dict


def zscore_all(data_dict, center=True):
    valid_scores = np.concatenate([x[~np.isnan(x).any(axis=1), :npcs] for x in data_dict.values()])
    mu, sig = valid_scores.mean(axis=0), valid_scores.std(axis=0)

    for k, v in data_dict.items():
        data_dict[k] = (v - mu) / sig

    return data_dict

# taken from syllables by @alewbw
def get_crosslikes(arhmm, frame_by_frame=False):
    '''Gets the cross-likelihoods, a measure of confidence in the model's
    segmentation, for each syllable a model learns.

    Args:
        arhmm: the ARHMM model object fit to your data
        frame_by_frame (bool): if True, the cross-likelihoods will be computed for
            each frame

    Returns:
        all_CLs: a dictionary containing cross-likelihoods for each syllable pair.
            if ``frame_by_frame=True``, it will contain a value for each frame
        CL: the average cross-likelihood for each syllable pair
    '''

    all_CLs = defaultdict(list)
    Nstates = arhmm.num_states

    if frame_by_frame:
        for s in arhmm.states_list:
            for j in range(Nstates):
                likes = s.aBl[s.stateseq == j]
                for i in range(Nstates):
                    all_CLs[(i, j)].append(likes[:, i] - likes[:, j])
        all_CLs = {k: np.concatenate(v) for k, v in all_CLs.items()}
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
    return [sl for sl in rleslices(indseq) if indseq[sl.start]]


def rleslices(seq):
    pos, = np.where(np.diff(seq) != 0)
    pos = np.concatenate(([0], pos+1, [len(seq)]))
    return map(slice, pos[:-1], pos[1:])
