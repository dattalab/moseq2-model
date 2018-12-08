import numpy as np
import joblib
import copy
import scipy.io
import h5py
from tqdm import tqdm_notebook, tqdm
from collections import OrderedDict


# grab matlab data
def load_pcs(filename, var_name="features", load_groups=False, npcs=10, h5_key_is_uuid=True):

    # TODO: trim pickles down to right number of pcs

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

        if isinstance(list(data_dict.values())[0], tuple):
            print('Detected tuple')
            for k, v in data_dict.items():
                data_dict[k] = v[0][:, :npcs]
                metadata['groups'].append(v[1])
        else:
            for k, v in data_dict.items():
                data_dict[k] = v[:, :npcs]

    elif filename.endswith('.h5'):
        with h5py.File(filename, 'r') as f:
            dsets = f.keys()

            if var_name in dsets:
                print('Found pcs in {}'.format(var_name))
                tmp = f[var_name]
                if isinstance(tmp, h5py._hl.dataset.Dataset):
                    data_dict = OrderedDict([(1, tmp.value[:, :npcs])])
                elif isinstance(tmp, h5py._hl.group.Group):
                    data_dict = OrderedDict([(k, v.value[:, :npcs]) for k, v in tmp.items()])
                    if 'groups' in dsets:
                        metadata['groups'] = [f['groups/{}'.format(key)].value for key in tmp.keys()]
                else:
                    raise IOError('Could not load data from h5 file')
            else:
                raise IOError('Could not find dataset name {} in {}'.format(var_name, filename))

            if 'uuids' in dsets:
                metadata['uuids'] = f['uuid'].value
            elif h5_key_is_uuid:
                metadata['uuids'] = list(data_dict.keys())

            # if 'groups' in dsets:
            #     print('Found groups in groups')
            #     metadata['groups'] = f['groups'].value

    else:
        raise ValueError('Did understand filetype')

    return data_dict, metadata


def save_dict(filename, obj_to_save=None):

    # we gotta switch to lists here my friend, create a file with multiple
    # pickles, only load as we need them

    if filename.endswith('.mat'):
        print('Saving MAT file '+filename)
        scipy.io.savemat(filename, mdict=obj_to_save)
    elif filename.endswith('.z'):
        print('Saving compressed pickle '+filename)
        joblib.dump(obj_to_save, filename, compress=3)
    elif filename.endswith('.pkl') | filename.endswith('.p'):
        print('Saving pickle '+filename)
        joblib.dump(obj_to_save, filename, compress=0)
    elif filename.endswith('.h5'):
        print('Saving h5 file '+filename)
        with h5py.File(filename, 'w') as f:
            recursively_save_dict_contents_to_group(f, obj_to_save)
    else:
        raise ValueError('Did understand filetype')


# https://codereview.stackexchange.com/questions/120802/recursively-save-python-dictionaries-to-hdf5-files-using-h5py
def recursively_save_dict_contents_to_group(h5file, export_dict, path='/'):
    """
    ....
    """
    for key, item in export_dict.items():
        if isinstance(key, (tuple, int)):
            key = str(key)

        if isinstance(item, str):
            item = item.encode('utf8')

        if isinstance(item, np.ndarray) and item.dtype == np.object:
            dt = h5py.special_dtype(vlen=item.flat[0].dtype)
            h5file.create_dataset(path+key, item.shape, dtype=dt, compression='gzip')
            for tup, idx in np.ndenumerate(item):
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


def load_arhmm_checkpoint(filename):
    return joblib.load(filename)


def save_arhmm_checkpoint(filename, arhmm):
    joblib.dump(arhmm, filename)


def load_dict_from_hdf5(filename):
    """
    ....
    """
    with h5py.File(filename, 'r') as h5file:
        return recursively_load_dict_contents_from_group(h5file, '/')


def recursively_load_dict_contents_from_group(h5file, path):
    """
    ....
    """
    ans = {}
    for key, item in h5file[path].items():
        if isinstance(item, h5py._hl.dataset.Dataset):
            ans[key] = item.value
        elif isinstance(item, h5py._hl.group.Group):
            ans[key] = recursively_load_dict_contents_from_group(h5file, path + key + '/')
    return ans


def load_data_from_matlab(filename, var_name="features", npcs=10):

    data_dict = OrderedDict()

    with h5py.File(filename, 'r') as f:
        if var_name in f.keys():
            score_tmp = f[var_name]
            for i in range(len(score_tmp)):
                tmp = f[score_tmp[i][0]]
                score_to_add = tmp.value
                data_dict[i] = score_to_add[:npcs, :].T

    return data_dict


def load_cell_string_from_matlab(filename, var_name="uuids"):

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
def copy_model(self):
    tmp = []

    # make a deep copy of the data-less version

    for s in self.states_list:
        tmp.append(s.data)
        s.data = None

    cp = copy.deepcopy(self)

    # now put the data back in

    for s, t in zip(self.states_list, tmp):
        s.data = t

    return cp


def get_parameters_from_model(model, save_ar=True):

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
        'num_states': trans_dist.N,
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
    cli = kwargs.pop('cli', False)

    if cli:
        return tqdm(*args, **kwargs)
    else:
        try:
            return tqdm_notebook(*args, **kwargs)
        except Exception:
            return tqdm(*args, **kwargs)


def list_rank(chk_list):
    rank = 0
    flag = True
    while flag is True:
        flag = eval("type(chk_list"+'[0]'*rank+") is list")
        if flag:
            rank += 1

    return rank


# https://stackoverflow.com/questions/2166818/python-how-to-check-if-an-object-is-an-instance-of-a-namedtuple
def is_named_tuple(variable):
    t = type(variable)
    b = t.__bases__

    if len(b) != 1 or b[0] != tuple:
        return False

    f = getattr(t, '_fields', None)

    if not isinstance(f, tuple):
        return False

    return all(type(n) == str for n in f)


# taken from moseq by @mattjj and @alexbw
def merge_dicts(base_dict, clobbering_dict):
    return dict(base_dict, **clobbering_dict)
