import os
import sys
import h5py
import numpy as np
import ruamel.yaml as yaml
from unittest import TestCase
from tests.unit_tests.test_train_utils import get_model
from moseq2_model.train.util import whiten_all, train_model
from autoregressive.models import FastARWeakLimitStickyHDPHMM
from moseq2_model.helpers.data import get_training_data_splits
from moseq2_model.util import load_data_from_matlab, load_cell_string_from_matlab, load_pcs, save_dict, dict_to_h5, \
                    append_resample, h5_to_dict, _load_h5_to_dict, copy_model, get_parameters_from_model, \
                    get_current_model, get_session_groupings, get_loglikelihoods

class TestUtils(TestCase):

    def test_get_current_model(self):

        use_checkpoint = False
        all_checkpoints = []

        test_model, data_dict = get_model()
        model_parameters = {
            'gamma': 1000,
            'alpha': 5.7,
            'kappa': 900,
            'nlags': 3,
            'robust': False,
            'max_states': 100,
            'separate_trans': False,
            'groups': None
        }

        arhmm, itr = get_current_model(use_checkpoint, all_checkpoints, data_dict, model_parameters)

        assert isinstance(arhmm, FastARWeakLimitStickyHDPHMM)
        assert itr == 0

    def test_get_session_groupings(self):

        input_data = 'data/test_scores.h5'
        data_dict, data_metadata = load_pcs(input_data, var_name='scores', load_groups=True)

        all_keys = list(data_dict.keys())

        groupings = get_session_groupings(data_metadata,
                                          ['default']*len(all_keys),
                                          all_keys,
                                          None)

        assert len(groupings) == 2
        assert groupings == ['default', 'default']

    def test_get_loglikelihoods(self):

        test_model, data_dict = get_model()

        nkeys = len(data_dict.keys())
        groups = ['default'] * nkeys
        separate_trans = False

        test_ll = get_loglikelihoods(test_model, data_dict, groups, separate_trans)

        assert len(test_ll) == nkeys

    def test_load_pcs(self):

        # first test loading dummy matlab file, then pickle, then h5
        pcs, metadata = load_pcs('data/dummy_matlab.mat', load_groups=True)

        assert(len(pcs) == 1)
        assert(len(metadata['groups']) == 1)
        assert(np.all(pcs[0] == 1))
        assert(metadata['groups'][0] == 'test')

        input_data = 'data/test_scores.h5'
        data_dict, data_metadata = load_pcs(input_data, var_name='scores', load_groups=True)

        assert list(data_dict.keys()) == data_metadata['uuids']

    def test_save_dict(self):
        input_data = 'data/test_scores.h5'
        data_dict, data_metadata = load_pcs(input_data, var_name='scores', load_groups=True)

        outfile = 'data/saved_dict.pkl'
        save_dict(outfile, data_dict)

        assert os.path.exists(outfile)
        os.remove(outfile)

    def test_append_resample(self):
        outfile = 'data/test_model.p'

        input_data = 'data/test_scores.h5'
        data_dict, data_metadata = load_pcs(input_data, var_name='scores', load_groups=True)

        append_resample(outfile, data_dict)

    def test_h5_to_dict(self):
        input_data = 'data/test_scores.h5'
        outdict = h5_to_dict(input_data, 'scores')
        assert isinstance(outdict, dict)

    def test_dict_to_h5(self):
        input_data = 'data/test_scores.h5'
        outdict = h5_to_dict(input_data, 'scores')
        assert isinstance(outdict, dict)

        outpath = 'data/out_scores.h5'

        tmp_h5 = h5py.File(outpath, 'w')
        dict_to_h5(tmp_h5, outdict)

        assert os.path.exists(outpath)
        os.remove(outpath)

    def test_load_h5_to_dict(self):
        input_data = 'data/test_scores.h5'
        with h5py.File(input_data, 'r') as f:
            outdict = _load_h5_to_dict(f, '/')
        assert isinstance(outdict, dict)

    def test_copy_model(self):
        model, data_dict = get_model()
        config_file = 'data/config.yaml'

        with open(config_file, 'r') as f:
            config_data = yaml.safe_load(f)

        X = whiten_all(data_dict)
        training_data, validation_data, nt_frames = get_training_data_splits(config_data, X)

        model, lls, labels, iter_lls, iter_holls, group_idx = train_model(model,
                                                                          num_iter=5, train_data=training_data,
                                                                          val_data=validation_data,
                                                                          num_frames=[900, 900])

        cp = copy_model(model)
        assert sys.getsizeof(model) == sys.getsizeof(cp)

    def test_get_parameters_from_model(self):

        def check_params(model, params):
            trans_dist = model.trans_distn
            assert params['kappa'] == trans_dist.kappa
            assert params['gamma'] == trans_dist.gamma
            assert params['max_states'] == trans_dist.N
            assert params['nlags'] == model.nlags
            assert params['ar_mat'] == [obs.A for obs in model.obs_distns]
            assert params['sig'] == [obs.sigma for obs in model.obs_distns]

        model, data_dict = get_model()
        config_file = 'data/config.yaml'

        with open(config_file, 'r') as f:
            config_data = yaml.safe_load(f)

        X = whiten_all(data_dict)
        training_data, validation_data, nt_frames = get_training_data_splits(config_data, X)

        model, lls, labels, iter_lls, iter_holls, group_idx = train_model(model,
                                                                          num_iter=5, train_data=training_data,
                                                                          val_data=validation_data,
                                                                          num_frames=[900, 900], separate_trans=True)

        params = get_parameters_from_model(model)
        check_params(model, params)

        model, lls, labels, iter_lls, iter_holls, group_idx = train_model(model,
                                                                          num_iter=5, train_data=training_data,
                                                                          val_data=validation_data,
                                                                          num_frames=[900, 900], separate_trans=False)

        params = get_parameters_from_model(model)
        check_params(model, params)

    def test_load_matlab_data(self):

        pcs = load_data_from_matlab('data/dummy_matlab.mat', var_name='features')
        keys = list(pcs.keys())

        assert(len(keys) == 1)
        assert(keys[0] == 0)
        assert(np.all(pcs[0] == 1))

    def test_load_cell_string_from_matlab(self):

        groups = load_cell_string_from_matlab('data/dummy_matlab.mat', var_name='groups')

        assert(len(groups) == 1)
        assert(groups[0] == 'test')
