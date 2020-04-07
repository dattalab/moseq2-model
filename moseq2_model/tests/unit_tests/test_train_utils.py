import ruamel.yaml as yaml
from unittest import TestCase
from moseq2_model.util import load_pcs
from moseq2_model.train.models import ARHMM
from moseq2_model.train.util import train_model, get_labels_from_model, whiten_all, whiten_each, \
                                    run_e_step
from autoregressive.models import FastARWeakLimitStickyHDPHMM, FastARWeakLimitStickyHDPHMMSeparateTrans
from moseq2_model.helpers.data import prepare_model_metadata, select_data_to_model, get_training_data_splits

def get_model(separate_trans=False, robust=False, groups=[]):
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

    model_parameters['separate_trans'] = separate_trans
    model_parameters['robust'] = robust
    model_parameters['groups'] = groups
    arhmm = ARHMM(data_dict=data_dict, **model_parameters)

    return arhmm, data_dict

class TestTrainUtils(TestCase):

    def test_train_model(self):
        config_file = 'data/config.yaml'
        with open(config_file, 'r') as f:
            config_data = yaml.safe_load(f)

        model, data_dict = get_model()

        X = whiten_all(data_dict)
        train_list, train_data, training_data, hold_out_list, validation_data, nt_frames = \
            get_training_data_splits(config_data, X)

        model, lls, labels, iter_lls, iter_holls, group_idx = train_model(model, save_file='data/out_model.p', num_iter=5, train_data=training_data,
                                                                          val_data=validation_data, num_frames=[900, 900])
        assert isinstance(model, FastARWeakLimitStickyHDPHMM)
        assert isinstance(lls, float)
        assert len(labels) == 2
        assert len(labels[0]) == 908
        assert len(iter_lls) == 0
        assert len(iter_holls) == 0
        assert len(group_idx) == 1
        assert group_idx == ['default']

        model, lls, labels, iter_lls, iter_holls, group_idx = train_model(model, save_file='data/out_model.p',
                                                                          num_iter=5, train_data=training_data,
                                                                          val_data=validation_data,
                                                                          num_frames=[900, 900], verbose=True)
        assert isinstance(model, FastARWeakLimitStickyHDPHMM)
        assert isinstance(lls, float)
        assert len(labels) == 2
        assert len(labels[0]) == 908
        assert len(iter_lls) == 5
        assert len(iter_holls) == 5
        assert len(group_idx) == 1
        assert group_idx == ['default']

        model, data_dict = get_model(separate_trans=True, groups=['default', 'Group1'])

        X = whiten_all(data_dict)
        train_list, train_data, training_data, hold_out_list, validation_data, nt_frames = \
            get_training_data_splits(config_data, X)

        model, lls, labels, iter_lls, iter_holls, group_idx = train_model(model, save_file='data/out_model.p',
                                                                          num_iter=5, train_data=training_data,
                                                                          val_data=validation_data, separate_trans=True,
                                                                          groups=['default', 'Group1'],
                                                                          num_frames=[900, 900], verbose=True)
        assert isinstance(model, FastARWeakLimitStickyHDPHMMSeparateTrans)
        assert isinstance(lls, float)
        assert len(labels) == 2
        assert len(labels[0]) == 908
        assert len(iter_lls) == 5
        assert len(iter_holls) == 5
        assert len(group_idx) == 2
        assert group_idx == ['default', 'Group1']

    def test_get_labels_from_model(self):

        config_file = 'data/config.yaml'

        with open(config_file, 'r') as f:
            config_data = yaml.safe_load(f)

        model, data_dict = get_model()

        X = whiten_all(data_dict)
        train_list, train_data, training_data, hold_out_list, validation_data, nt_frames = \
            get_training_data_splits(config_data, X)

        model, lls, labels, iter_lls, iter_holls, group_idx = train_model(model, save_file='data/out_model.p',
                                                                          num_iter=5, train_data=training_data,
                                                                          val_data=validation_data,
                                                                          num_frames=[900, 900])

        labels = get_labels_from_model(model)
        print(labels)
        assert len(labels[0]) == 908

    def test_whiten_all(self):

        _, data_dict = get_model()

        whitened_a = whiten_all(data_dict)
        whitened_e = whiten_each(data_dict)
        assert data_dict.values() != whitened_a.values()
        assert whitened_a.values() != whitened_e.values()

    def test_whiten_each(self):
        _, data_dict = get_model()

        whitened_a = whiten_all(data_dict,center=False)
        whitened_e = whiten_each(data_dict,center=False)
        assert data_dict.values() != whitened_a.values()
        assert whitened_a.values() != whitened_e.values()

    def test_run_estep(self):
        model, _ = get_model()
        ex_states = run_e_step(model)
        assert len(ex_states) == 2
        assert len(ex_states[0]) == 905 # 908 - 3 nlag frames
