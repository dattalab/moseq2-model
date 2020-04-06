import ruamel.yaml as yaml
from unittest import TestCase
from moseq2_model.util import load_pcs
from moseq2_model.train.models import ARHMM
from moseq2_model.helpers.data import prepare_model_metadata, select_data_to_model
from autoregressive.models import FastARWeakLimitStickyHDPHMM, FastARWeakLimitStickyHDPHMMSeparateTrans, \
                            ARWeakLimitStickyHDPHMM, ARWeakLimitStickyHDPHMMSeparateTrans

class TestTrainModels(TestCase):

    def test_get_empirical_ar_params(self):
        print()

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
