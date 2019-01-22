import os
import shutil
import numpy as np
from functools import partial
from collections import OrderedDict, defaultdict
from moseq2_model.util import progressbar, save_arhmm_checkpoint


# based on moseq by @mattjj and @alexbw
def train_model(model, num_iter=100, save_every=1, ncpus=1, cli=False, **kwargs):

    # per conversations w/ @mattjj, the fast class of models use openmp no need
    # for "extra" parallelism

    log_likelihoods = kwargs.pop('log_likelihoods', [])
    labels = kwargs.pop('labels', [])

    save_progress = kwargs.pop('save_progress', None)
    filename = kwargs.pop('filename', 'model.arhmm')
    filename = os.path.splitext(filename)[0] + '-checkpoint.arhmm'
    start = kwargs.pop('iter', 0)

    for itr in progressbar(range(start, num_iter), cli=cli, **kwargs):
        model.resample_model(num_procs=ncpus)
        if (np.mod(itr+1, save_every) == 0 or
                np.mod(itr+1, num_iter) == 0):
            log_likelihoods.append(model.log_likelihood())
            seq_list = [s.stateseq for s in model.states_list]
            for seq_itr in range(len(seq_list)):
                seq_list[seq_itr] = np.append(np.repeat(-5, model.nlags), seq_list[seq_itr])
            labels.append(seq_list)
        if save_progress is not None and (itr + 1) % save_progress == 0:
            # move around the checkpoints
            if os.path.exists(filename):
                if os.path.exists(filename + '.1'):
                    os.remove(filename + '.1')
                shutil.move(filename, filename + '.1')
            save_arhmm_checkpoint(filename, {'iter': itr, 'model': model,
                'log_likelihoods': log_likelihoods, 'labels': labels})

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


# taken from syllables by @alewbw
def get_crosslikes(arhmm, frame_by_frame=False):
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
    return [sl for sl in rleslices(indseq) if indseq[sl.start]]


def rleslices(seq):
    pos, = np.where(np.diff(seq) != 0)
    pos = np.concatenate(([0], pos+1, [len(seq)]))
    return map(slice, pos[:-1], pos[1:])
