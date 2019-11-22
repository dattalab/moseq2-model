import os
import numpy as np
from functools import partial
from collections import OrderedDict, defaultdict
from moseq2_model.util import progressbar, save_arhmm_checkpoint, append_resample

def train_model(model, num_iter=100, save_every=1, ncpus=1, checkpoint_freq=None,
                checkpoint_file=None, start=0, save_file=None, progress_kwargs={}, num_frames=[1], val_data=None, separate_trans=False):

    checkpoint = checkpoint_freq is not None

    iter_lls = []
    iter_holls = []
    for itr in progressbar(range(start, num_iter), **progress_kwargs):
        try:
            model.resample_model(num_procs=ncpus)
            if not separate_trans:
                train_ll = model.log_likelihood()/sum(num_frames)
                print(train_ll)
                iter_lls.append(train_ll)
                '''
            else:
                train_ll = [model.log_likelihood(v, group_id=g) for g, v in zip(groups, train_data.values())]
                print(train_ll)
                print(get_labels_from_model(model))
                iter_lls.append(train_ll)
                '''
            if val_data is not None:
                if not separate_trans:
                    val_ll = sum([(model.log_likelihood(v)/len(v)) for v in val_data.values()])/len(val_data.values())
                    print(val_ll)
                    iter_holls.append(val_ll)
                    '''
                else:
                    val_ll = [model.log_likelihood(v, group_id=g) for g, v in zip(groups, val_data.values())]
                    print(val_ll)
                    print(get_labels_from_model(model))
                    iter_holls.append(val_ll)
                    '''
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
                # move around the checkpoints
                if os.path.exists(checkpoint_file):
                    checkpoint_file = checkpoint_file + '.1'
                save_arhmm_checkpoint(checkpoint_file, save_data)
        except:
            break

    return model, model.log_likelihood(), get_labels_from_model(model), iter_lls, iter_holls


def get_labels_from_model(model):
    '''grabs the model labels for each training dataset and places them in a list'''
    cat_labels = [np.append(np.repeat(-5, model.nlags), s.stateseq) for s in model.states_list]
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

def run_e_step(arhmm):
    '''computes the expected states for each training dataset and places them in a list'''
    arhmm._E_step()
    return [s.expected_states for s in arhmm.states_list]


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
