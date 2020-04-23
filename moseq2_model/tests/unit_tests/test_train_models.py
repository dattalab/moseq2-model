import numpy as np
import ruamel.yaml as yaml
from unittest import TestCase
from moseq2_model.util import load_pcs
from moseq2_model.train.models import ARHMM, _get_empirical_ar_params
from moseq2_model.helpers.data import prepare_model_metadata, select_data_to_model
from autoregressive.models import FastARWeakLimitStickyHDPHMM, FastARWeakLimitStickyHDPHMMSeparateTrans, \
                            ARWeakLimitStickyHDPHMM, ARWeakLimitStickyHDPHMMSeparateTrans

class TestTrainModels(TestCase):

    def test_get_empirical_ar_params(self):
        input_file = 'data/test_scores.h5'

        data_dict, data_metadata = load_pcs(filename=input_file,
                                            var_name='scores',
                                            npcs=10,
                                            load_groups=True)

        data_dim = list(data_dict.values())[0].shape[1]
        S_0_scale = 0.01
        K_0_scale = 10.0
        affine = True
        nlags = 3

        params = {
            'nu_0': data_dim + 2,
            'S_0': S_0_scale * np.eye(data_dim),
            'M_0': np.hstack((np.eye(data_dim),
                              np.zeros((data_dim, data_dim * (nlags - 1))),
                              np.zeros((data_dim, int(affine))))),
            'affine': affine,
            'K_0': K_0_scale * np.eye(data_dim * nlags + affine)
        }

        obs_params = dict(nu_0=params["nu_0"],
                          S_0=params['S_0'],
                          M_0=params['M_0'],
                          K_0=params['K_0'],
                          affine=params['affine'])

        new_params = _get_empirical_ar_params(list(data_dict.values()), obs_params)
        assert len(new_params['S_0']) == 10 # npcs
        assert len(new_params['K_0']) == len(params['K_0'])
        assert len(new_params['M_0']) == len(params['M_0'])
        assert new_params['affine'] == params['affine']

    def test_ARHMM(self):
        input_file = 'data/test_scores.h5'
        config_file = 'data/config.yaml'
        index_path = 'data/test_index.yaml'

        with open(config_file, 'r') as f:
            config_data = yaml.safe_load(f)

        with open(index_path, 'r') as f:
            index_data = yaml.safe_load(f)
        f.close()

        all_keys, groups = select_data_to_model(index_data)

        data_dict, data_metadata = load_pcs(filename=input_file,
                                            var_name='scores',
                                            npcs=10,
                                            load_groups=True)

        config_data, data_dict, model_parameters, train_list, hold_out_list = \
            prepare_model_metadata(data_dict, data_metadata, config_data, \
                                    len(all_keys), all_keys)

        model_parameters['separate_trans'] = False
        model_parameters['robust'] = False
        arhmm = ARHMM(data_dict=data_dict, **model_parameters)
        assert isinstance(arhmm, FastARWeakLimitStickyHDPHMM)

        model_parameters['sticky_init'] = True
        arhmm = ARHMM(data_dict=data_dict, **model_parameters)
        assert isinstance(arhmm, FastARWeakLimitStickyHDPHMM)
        model_parameters['sticky_init'] = False

        model_parameters['separate_trans'] = True
        model_parameters['robust'] = False
        model_parameters['groups'] = ['1', '2']
        arhmm = ARHMM(data_dict=data_dict, **model_parameters)
        assert isinstance(arhmm, FastARWeakLimitStickyHDPHMMSeparateTrans)

        model_parameters['separate_trans'] = False
        model_parameters['robust'] = True
        arhmm = ARHMM(data_dict=data_dict, **model_parameters)
        assert isinstance(arhmm, ARWeakLimitStickyHDPHMM)

        model_parameters['separate_trans'] = True
        model_parameters['robust'] = True
        arhmm = ARHMM(data_dict=data_dict, **model_parameters)
        assert isinstance(arhmm, ARWeakLimitStickyHDPHMMSeparateTrans)