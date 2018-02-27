from __future__ import division
from __future__ import print_function
import numpy as np
import h5py as h5
import joblib
import copy
import ruamel.yaml as yaml
import itertools
import hdf5storage
from tqdm import tqdm_notebook, tqdm
from collections import OrderedDict


# grab matlab data
def load_pcs(filename, var_name="features", load_groups=False, npcs=10):

    # TODO: trim pickles down to right number of pcs

    metadata = {
        'uuids': None,
        'groups': None
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
        data_dict = joblib.load(filename)
    elif filename.endswith('.h5'):
        with h5.File(filename, 'r') as f:
            dsets = f.keys()
            print('In {} found {} datasets'.format(filename, dsets))
            if var_name in dsets:
                print('Found pcs in {}'.format(var_name))
                tmp = f[var_name].value
                if type(tmp) is np.ndarray:
                    data_dict = OrderedDict([(1, tmp)])
                elif type(tmp) is dict:
                    data_dict = OrderedDict([(k, v) for k, v in dict.iteritems()])
                elif type(tmp) is OrderedDict:
                    data_dict = tmp

            if 'uuid' in dsets:
                print('Found groups in groups')
                metadata['uuid'] = f['uuid'].value

            if 'groups' in dsets:
                print('Found groups in groups')
                metadata['groups'] = f['groups'].value

    else:
        raise ValueError('Did understand filetype')

    return data_dict, metadata


def save_dict(filename, obj_to_save=None):

    # we gotta switch to lists here my friend, create a file with multiple
    # pickles, only load as we need them

    if filename.endswith('.mat'):
        print('Saving MAT file '+filename)
        sio.savemat(filename, mdict=obj_to_save)
    elif filename.endswith('.z'):
        print('Saving compressed pickle '+filename)
        joblib.dump(obj_to_save, filename, compress=3)
    elif filename.endswith('.pkl') | filename.endswith('.p'):
        print('Saving pickle '+filename)
        joblib.dump(obj_to_save, filename, compress=0)
    elif filename.endswith('.h5'):
        print('Saving h5 file '+filename)
        with h5.File(filename, 'w') as f:
            recursively_save_dict_contents_to_group(f, obj_to_save)
    else:
        raise ValueError('Did understand filetype')


# https://codereview.stackexchange.com/questions/120802/recursively-save-python-dictionaries-to-hdf5-files-using-h5py
def recursively_save_dict_contents_to_group(h5file, export_dict, path='/'):
    """
    ....
    """
    for key, item in export_dict.iteritems():
        if isinstance(key, (tuple, int)):
            key = str(key)

        if isinstance(item, unicode):
            item = item.encode('utf8')

        if isinstance(item, np.ndarray) and item.dtype == np.object:
            dt = h5.special_dtype(vlen=item.flat[0].dtype)
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


def load_dict_from_hdf5(filename):
    """
    ....
    """
    with h5.File(filename, 'r') as h5file:
        return recursively_load_dict_contents_from_group(h5file, '/')


def recursively_load_dict_contents_from_group(h5file, path):
    """
    ....
    """
    ans = {}
    for key, item in h5file[path].items():
        if isinstance(item, h5._hl.dataset.Dataset):
            ans[key] = item.value
        elif isinstance(item, h5._hl.group.Group):
            ans[key] = recursively_load_dict_contents_from_group(h5file, path + key + '/')
    return ans


def load_data_from_matlab(filename, var_name="features", npcs=10):

    data_dict = OrderedDict()

    with h5.File(filename, 'r') as f:
        if var_name in f.keys():
            score_tmp = f[var_name]
            for i in range(len(score_tmp)):
                tmp = f[score_tmp[i][0]]
                score_to_add = tmp.value
                data_dict[i] = score_to_add[:npcs, :].T

    return data_dict


def load_cell_string_from_matlab(filename, var_name="uuids"):

    f = h5.File(filename)
    return_list = []

    if var_name in f.keys():

        tmp = f[var_name]

        for i in xrange(len(tmp)):
            tmp2 = f[tmp[i][0]]
            uni_list = [''.join(unichr(c)) for c in tmp2]
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
        parameters['nu'] = model.obs_distns[0].nu

    if save_ar:
        parameters['ar_mat'] = [obs.A for obs in model.obs_distns]
        parameters['sig'] = [obs.sigma for obs in model.obs_distns]

    return parameters


# read in user yml file for mpi jobs
def read_cli_config(filename, suppress_output=False):

    with open(filename, 'r') as f:
        config = yaml.load(f.read(), Loader=yaml.Loader)

    cfg = {
        'worker_dicts': None,
        'scan_parameter': None,
        'scan_values': None,
        'other_parameters': {}
    }

    if 'scan_settings' in config:

        cfg = merge_dicts(cfg, config['scan_settings'])
        cfg['scan_values'] = []
        cfg['worker_dicts'] = []

        if type(cfg['scan_parameter']) is list:
            for use_parameter, use_range, use_scale in zip(cfg['scan_parameter'],
                                                           cfg['scan_range'], cfg['scan_scale']):
                if use_scale == 'log':
                    cfg['scan_values'].append(np.logspace(*use_range))
                elif use_scale == 'linear':
                    cfg['scan_values'].append(np.linspace(*use_range))
                else:
                    cfg['scan_values'].append(use_range)

            for itr_values in itertools.product(*cfg['scan_values']):
                new_dict = {}
                for param, value in zip(cfg['scan_parameter'], itr_values):
                    new_dict[param] = value
                cfg['worker_dicts'].append(new_dict)
        else:
            if cfg['scan_scale'] == 'log':
                cfg['scan_values'].append(np.logspace(*cfg['scan_range']))
            elif cfg['scan_scale'] == 'linear':
                cfg['scan_values'].append(np.linspace(*cfg['scan_range']))

            for value in cfg['scan_values'][0]:
                new_dict = {
                    cfg['scan_parameter']: value
                }
                cfg['worker_dicts'].append(new_dict)

    cfg['other_parameters'] = {}

    if 'parameters' in config.keys():
        cfg['other_parameters'] = config['parameters']

    if not suppress_output:
        if type(cfg['scan_parameter']) is list:
            for param, values in zip(cfg['scan_parameter'], cfg['scan_values']):
                print('Will scan parameter '+param)
                print('Will scan value '+str(values))
        else:
            print('Will scan parameter '+cfg['scan_parameter'])
            print('Will scan value '+str(cfg['scan_values'][0]))

    return cfg


# credit to http://stackoverflow.com/questions/14000893/specifying-styles-for-portions-of-a-pyyaml-dump
# class blockseq(dict):
#     pass
#
#
# def blockseq_rep(dumper, data):
#     return dumper.represent_mapping(u'tag:yaml.org,2002:map', data, flow_style=False)
#
#
# class flowmap(dict):
#     pass
#
#
# def flowmap_rep(dumper, data):
#     return dumper.represent_mapping(u'tag:yaml.org,2002:map', data, flow_style=True)


# from http://stackoverflow.com/questions/16782112/can-pyyaml-dump-dict-items-in-non-alphabetical-order
# def represent_ordereddict(dumper, data):
#     value = []
#
#     for item_key, item_value in data.items():
#         node_key = dumper.represent_data(item_key)
#         node_value = dumper.represent_data(item_value)
#
#         value.append((node_key, node_value))
#
#     return yaml.nodes.MappingNode(u'tag:yaml.org,2002:map', value)


# taken from moseq by @mattjj and @alexbw
def merge_dicts(base_dict, clobbering_dict):
    return dict(base_dict, **clobbering_dict)


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