import os
import pickle
import numpy as np
import joblib
import scipy.io
import h5py
from copy import deepcopy
from cytoolz import first
from functools import partial
from tqdm.auto import tqdm
from collections import OrderedDict
from autoregressive.util import AR_striding

flush_print = partial(print, flush=True)

def load_pcs(filename, var_name="features", load_groups=False, npcs=10, h5_key_is_uuid=True):
    '''
    Load the Principal Component Scores for modeling.

    Parameters
    ----------
    filename (str): path to the file that contains PC scores
    var_name (str): key where the pc scores are stored within ``filename``
    load_groups (bool): Load metadata group variable
    npcs (int): Number of PCs to load
    h5_key_is_uuid (bool): use h5 key as uuid.

    Returns
    -------
    data_dict (OrderedDict): key-value pairs for keys being uuids and values being PC scores.
    metadata (OrderedDict): dictionary containing lists of index-aligned uuids and groups.
    '''

    metadata = {
        'uuids': None,
        'groups': [],
    }

    if filename.endswith('.mat'):
        print('Loading data from matlab file')
        data_dict = load_data_from_matlab(filename, var_name, npcs)
        # convert the uuid list to something that will export easily...
        metadata['uuids'] = load_cell_string_from_matlab(filename, "uuids")
        if load_groups:
            metadata['groups'] = load_cell_string_from_matlab(filename, "groups")
        else:
            metadata['groups'] = None
    elif filename.endswith('.z') or filename.endswith('.pkl') or filename.endswith('.p'):
        print('Loading data from pickle file')
        data_dict = joblib.load(filename)

        if isinstance(first(data_dict.values()), tuple):
            print('Detected tuple')
            for k, v in data_dict.items():
                data_dict[k] = v[0][:, :npcs]
                metadata['groups'].append(v[1])
        else:
            for k, v in data_dict.items():
                data_dict[k] = v[:, :npcs]

    elif filename.endswith('.h5'):
        with h5py.File(filename, 'r') as f:
            if var_name in f:
                print('Found pcs in {}'.format(var_name))
                tmp = f[var_name]
                if isinstance(tmp, h5py.Dataset):
                    data_dict = OrderedDict([(1, tmp[:, :npcs])])
                elif isinstance(tmp, h5py.Group):
                    data_dict = OrderedDict([(k, v[:, :npcs]) for k, v in tmp.items()])
                    if load_groups:
                        metadata['groups'] = list(range(len(tmp)))
                    elif 'groups' in f:
                        metadata['groups'] = [f[f'groups/{key}'][()] for key in tmp.keys()]

                else:
                    raise IOError('Could not load data from h5 file')
            else:
                raise IOError(f'Could not find dataset name {var_name} in {filename}')

            if 'uuids' in f:
                # TODO: make sure uuids is in f, and not uuid
                metadata['uuids'] = f['uuid'][()]
            elif h5_key_is_uuid:
                metadata['uuids'] = list(data_dict.keys())

    else:
        raise ValueError('Did not understand filetype')

    return data_dict, metadata


def save_dict(filename, obj_to_save=None):
    '''
    Save dictionary to file.

    Parameters
    ----------
    filename (str): path to file where dict is being saved.
    obj_to_save (dict): dict to save.

    Returns
    -------
    None
    '''

    # we gotta switch to lists here my friend, create a file with multiple
    # pickles, only load as we need them

    if filename.endswith('.mat'):
        print('Saving MAT file', filename)
        scipy.io.savemat(filename, mdict=obj_to_save)
    elif filename.endswith('.z'):
        print('Saving compressed pickle', filename)
        joblib.dump(obj_to_save, filename, compress=('zlib', 4))
    elif filename.endswith('.pkl') | filename.endswith('.p'):
        print('Saving pickle', filename)
        joblib.dump(obj_to_save, filename, compress=0)
    elif filename.endswith('.h5'):
        print('Saving h5 file', filename)
        with h5py.File(filename, 'w') as f:
            # TODO: rename this function to be consistent with the other repos
            recursively_save_dict_contents_to_group(f, obj_to_save)
    else:
        raise ValueError('Did not understand filetype')



def recursively_save_dict_contents_to_group(h5file, export_dict, path='/'):
    '''
    Recursively save dicts to h5 file groups.
    # https://codereview.stackexchange.com/questions/120802/recursively-save-python-dictionaries-to-hdf5-files-using-h5py

    Parameters
    ----------
    h5file (h5py.File): opened h5py File object.
    export_dict (dict): dictionary to save
    path (str): path within h5 to save to.

    Returns
    -------
    None
    '''

    for key, item in export_dict.items():
        if isinstance(key, (tuple, int)):
            key = str(key)

        if isinstance(item, str):
            item = item.encode('utf8')

        if isinstance(item, np.ndarray) and item.dtype == np.object:
            dt = h5py.special_dtype(vlen=item.flat[0].dtype)
            h5file.create_dataset(path+key, item.shape, dtype=dt, compression='gzip')
            for tup, _ in np.ndenumerate(item):
                if item[tup] is not None:
                    h5file[path+key][tup] = np.array(item[tup]).ravel()
        elif isinstance(item, (np.ndarray, list)):
            h5file.create_dataset(path+key, data=item, compression='gzip')
        elif isinstance(item, (np.int, np.float, str, bytes)):
            h5file.create_dataset(path+key, data=item)
        elif isinstance(item, dict):
            recursively_save_dict_contents_to_group(h5file, item, path + key + '/')
        else:
            raise ValueError('Cannot save {} type'.format(type(item)))


def load_arhmm_checkpoint(filename: str, train_data: dict) -> dict:
    '''
    Load an arhmm checkpoint and re-add data into the arhmm model checkpoint.

    Parameters
    ----------
    filename (str): path that specifies the checkpoint.
    train_data (OrderedDict): an OrderedDict that contains the training data

    Returns
    -------
    mdl_dict (dict): a dict containing the model with reloaded data, and associated training data
    '''

    mdl_dict = joblib.load(filename)
    nlags = mdl_dict['model'].nlags

    for s, t in zip(mdl_dict['model'].states_list, train_data.values()):
        s.data = AR_striding(t.astype('float32'), nlags)

    return mdl_dict

def save_arhmm_checkpoint(filename: str, arhmm: dict):
    '''
    Save an arhmm checkpoint and strip out data used to train the model.

    Parameters
    ----------
    filename (str): path that specifies the checkpoint
    arhmm (dict): a dictionary containing the model obj, training iteration number,
               log-likelihoods of each training step, and labels for each step.

    Returns
    -------
    None
    '''

    mdl = arhmm.pop('model')
    arhmm['model'] = copy_model(mdl)
    joblib.dump(arhmm, filename, compress=('zlib', 5))


def append_resample(filename, label_dict: dict):
    '''
    Adds the labels from a resampling iteration to a pickle file.

    Parameters
    ----------
    filename (str): file (containing modeling results) to append new label dict to.
    label_dict (dict): a dictionary with a single key/value pair, where the
            key is the sampling iteration and the value contains a dict of:
            (labels, a log likelihood val, and expected states if the flag is set)
            from each mouse.

    Returns
    -------
    None
    '''

    with open(filename, 'ab+') as f:
        pickle.dump(label_dict, f)


def load_dict_from_hdf5(filename):
    '''
    A convenience function to load the entire contents of an h5 file
    into a dictionary.

    Parameters
    ----------
    filename (str): path to h5 file.

    Returns
    -------
    (dict): dict containing all of the h5 file contents.
    '''

    return h5_to_dict(filename, '/')


def _load_h5_to_dict(file: h5py.File, path: str) -> dict:
    '''
    A convenience function to load the contents of an h5 file
    at a user-specified path into a dictionary.

    Parameters
    ----------
    filename (str): path to h5 file.
    path (str): path within the h5 file to load data from.

    Returns
    -------
    (dict): dict containing all of the h5 file contents.
    '''

    ans = {}
    if isinstance(file[path], h5py._hl.dataset.Dataset):
        # only use the final path key to add to `ans`
        ans[path.split('/')[-1]] = file[path][()]
    else:
        for key, item in file[path].items():
            if isinstance(item, h5py.Dataset):
                ans[key] = item[()]
            elif isinstance(item, h5py.Group):
                ans[key] = _load_h5_to_dict(file, '/'.join([path, key]))
    return ans


def h5_to_dict(h5file, path: str) -> dict:
    '''
    Load h5 data to dictionary from a user specified path.

    Parameters
    ----------
    h5file (str or h5py.File): file path to the given h5 file or the h5 file handle
    path (str): path to the base dataset within the h5 file

    Returns
    -------
    out (dict): a dict with h5 file contents with the same path structure
    '''

    if isinstance(h5file, str):
        with h5py.File(h5file, 'r') as f:
            out = _load_h5_to_dict(f, path)
    elif isinstance(h5file, (h5py.File, h5py.Group)):
        out = _load_h5_to_dict(h5file, path)
    else:
        raise Exception('file input not understood - need h5 file path or file object')
    return out


def load_data_from_matlab(filename, var_name="features", npcs=10):
    '''
    Load PC Scores from a specified variable column in a MATLAB file.

    Parameters
    ----------
    filename (str): path to MATLAB (.mat) file
    var_name (str): variable to load
    npcs (int): number of PCs to load.

    Returns
    -------
    data_dict (OrderedDict): loaded dictionary of uuid and PC-score pairings.
    '''

    data_dict = OrderedDict()

    with h5py.File(filename, 'r') as f:
        if var_name in f.keys():
            score_tmp = f[var_name]
            for i in range(len(score_tmp)):
                tmp = f[score_tmp[i][0]]
                score_to_add = tmp[()]
                data_dict[i] = score_to_add[:npcs, :].T

    return data_dict


def load_cell_string_from_matlab(filename, var_name="uuids"):
    '''
    Load cell strings from MATLAB file.

    Parameters
    ----------
    filename (str): path to .mat file
    var_name (str): cell name to read

    Returns
    -------
    return_list (list): list of selected loaded variables
    '''

    f = h5py.File(filename)
    return_list = []

    if var_name in f.keys():

        tmp = f[var_name]

        # change unichr to chr for python 3

        for i in range(len(tmp)):
            tmp2 = f[tmp[i][0]]
            uni_list = [''.join(chr(c)) for c in tmp2]
            return_list.append(''.join(uni_list))

    return return_list


# per Scott's suggestion
def copy_model(model_obj):
    '''
    Return a new copy of a model using deepcopy().

    Parameters
    ----------
    model_obj (ARHMM): model to copy.

    Returns
    -------
    cp (ARHMM): copy of the model
    '''

    tmp = []

    # make a deep copy of the data-less version
    for s in model_obj.states_list:
        tmp.append(s.data)
        s.data = None

    cp = deepcopy(model_obj)

    # now put the data back in

    for s, t in zip(model_obj.states_list, tmp):
        s.data = t

    return cp


def get_parameters_from_model(model, save_ar=True):
    '''
    Get parameter dictionary from model.

    Parameters
    ----------
    model (ARHMM): model to get parameters from.
    save_ar (bool): save AR Matrices.

    Returns
    -------
    parameters (dict): dictionary containing all modeling parameters
    '''

    # trans_dist=model.trans_distn
    init_obs_dist = model.init_emission_distn.hypparams

    # need to be smarter about this, but for now assume parameters are the same
    # (eek!) if we use separate trans

    try:
        trans_dist = model.trans_distn
    except Exception:
        tmp = model.trans_distns
        trans_dist = tmp[0]

    ls_obj = dir(model.obs_distns[0])

    parameters = {
        'kappa': trans_dist.kappa,
        'gamma': trans_dist.gamma,
        'alpha': trans_dist.alpha,
        'nu': np.nan,
        'max_states': trans_dist.N,
        'nu_0': init_obs_dist['nu_0'],
        'sigma_0': init_obs_dist['sigma_0'],
        'kappa_0': init_obs_dist['kappa_0'],
        'nlags': model.nlags,
        'mu_0': init_obs_dist['mu_0'],
        'model_class': model.__class__.__name__
        }

    if 'nu' in ls_obj:
        parameters['nu'] = [obs.nu for obs in model.obs_distns]

    if save_ar:
        parameters['ar_mat'] = [obs.A for obs in model.obs_distns]
        parameters['sig'] = [obs.sigma for obs in model.obs_distns]

    return parameters


def progressbar(*args, **kwargs):
    '''
    Selects tqdm progress bar.

    Parameters
    ----------
    args (iterable)
    kwargs (tdqm args[1:])

    Returns
    -------
    tqdm() iterating object.
    '''

    cli = kwargs.pop('cli', False)

    if cli:
        return tqdm(*args, **kwargs)
    else:
        return tqdm(*args, **kwargs)


def list_rank(chk_list):
    rank = 0
    flag = True
    while flag is True:
        flag = eval("type(chk_list"+'[0]'*rank+") is list")
        if flag:
            rank += 1

    return rank