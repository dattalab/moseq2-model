import os
import sys
import ruamel.yaml as yaml
from unittest import TestCase
from moseq2_model.util import load_pcs
from moseq2_model.helpers.data import process_indexfile, select_data_to_model, prepare_model_metadata,\
    get_heldout_data_splits, get_training_data_splits, graph_modeling_loglikelihoods

class TestDataHelpers(TestCase):

    def test_process_indexfile(self):

        input_file = 'data/test_scores.h5'
        index_path = 'data/test_index.yaml'
        config_file = 'data/config.yaml'

        data_dict, data_metadata = load_pcs(filename=input_file,
                                            var_name='scores',
                                            npcs=10,
                                            load_groups=True)

        with open(config_file, 'r') as f:
            config_data = yaml.safe_load(f)

        index_data, data_metadata = process_indexfile(index_path, config_data, data_metadata)
        assert (len(index_data['files']) == len(data_metadata['uuids'])),\
            "Number of input files != number of uuids in the index file"

    def test_select_data_to_model(self):
        index_path = 'data/test_index.yaml'

        with open(index_path, 'r') as f:
            index_data = yaml.safe_load(f)

        # test simple CLI case
        all_keys, groups = select_data_to_model(index_data)
        assert len(all_keys) == len(groups), "Number of groups != number of uuids"

        # test single GUI input
        index_data['files'][1]['group'] = 'default'
        stdin = 'data/stdin.txt'
        with open(stdin, 'w') as f:
            f.write('default')

        sys.stdin = open(stdin)
        all_keys, groups = select_data_to_model(index_data, select_groups=True)

        assert len(all_keys) == len(groups) == 1, "index data was incorrectly parsed"
        assert groups[0] == 'default', "groups were returned incorrectly"

        # test space-separated input
        with open(stdin, 'w') as f:
            f.write('default Group1')

        sys.stdin = open(stdin)
        all_keys, groups = select_data_to_model(index_data, select_groups=True)

        assert len(all_keys) == len(groups) == 2, "index data was incorrectly parsed"
        self.assertCountEqual(groups, ['default', 'Group1'], "groups were returned incorrectly")

        # test comma-separated input
        with open(stdin, 'w') as f:
            f.write('default, Group1')

        sys.stdin = open(stdin)
        all_keys, groups = select_data_to_model(index_data, select_groups=True)

        assert len(all_keys) == len(groups) == 2, "index data was incorrectly parsed"
        self.assertCountEqual(groups, ['default', 'Group1'], "groups were returned incorrectly")
        os.remove(stdin)

    def test_prepare_model_metadata(self):

        input_file = 'data/test_scores.h5'
        config_file = 'data/config.yaml'

        data_dict, data_metadata = load_pcs(filename=input_file,
                                            var_name='scores',
                                            npcs=10,
                                            load_groups=True)

        with open(config_file, 'r') as f:
            config_data = yaml.safe_load(f)

        index_path = 'data/test_index.yaml'

        with open(index_path, 'r') as f:
            index_data = yaml.safe_load(f)

        all_keys, groups = select_data_to_model(index_data)

        config_data, data_dict1, model_parameters, train_list, hold_out_list = \
            prepare_model_metadata(data_dict, data_metadata, config_data, len(all_keys), all_keys)

        assert data_dict.values() != data_dict1.values(), "Index loaded uuids and training data does not match scores file"
        assert train_list == all_keys, "Loaded uuids do not match total number of uuids"
        assert hold_out_list == None, "Some of the data is unintentionally held out"

        config_data['whiten'] = 'each'
        config_data['noise_level'] = 1
        config_data, data_dict1, model_parameters, train_list, hold_out_list = \
            prepare_model_metadata(data_dict, data_metadata, config_data, len(all_keys), all_keys)

        assert data_dict.values() != data_dict1.values(), "Index loaded uuids and training data does not match scores file"
        assert train_list == all_keys, "Loaded uuids do not match total number of uuids"
        assert hold_out_list == None, "Some of the data is unintentionally held out"

        config_data['whiten'] = 'none'
        config_data, data_dict1, model_parameters, train_list, hold_out_list = \
            prepare_model_metadata(data_dict, data_metadata, config_data, len(all_keys), all_keys)

        assert data_dict.values() != data_dict1.values(), "Index loaded uuids and training data does not match scores file"
        assert train_list == all_keys, "Loaded uuids do not match total number of uuids"
        assert hold_out_list == None, "Some of the data is unintentionally held out"

    def test_get_heldout_data_splits(self):
        input_file = 'data/test_scores.h5'
        config_file = 'data/config.yaml'

        data_dict, data_metadata = load_pcs(filename=input_file,
                                            var_name='scores',
                                            npcs=10,
                                            load_groups=True)

        with open(config_file, 'r') as f:
            config_data = yaml.safe_load(f)
            config_data['hold_out'] = True
            config_data['nfolds'] = 2

        index_path = 'data/test_index.yaml'

        with open(index_path, 'r') as f:
            index_data = yaml.safe_load(f)

        all_keys, groups = select_data_to_model(index_data)

        config_data, data_dict, model_parameters, train_list, hold_out_list = \
            prepare_model_metadata(data_dict, data_metadata, config_data, len(all_keys), all_keys)

        assert (sorted(train_list) != sorted(hold_out_list)), "Training data is the same as held out data"
        assert (len(train_list) == len(hold_out_list)), "Number of held out sets is incorrect, supposed to be 1"

        train_data, hold_out_list, test_data, nt_frames = \
            get_heldout_data_splits(all_keys, data_dict, train_list, hold_out_list)

        assert(train_list != None and test_data != None), "There are missing datasets"


    def test_get_training_data_splits(self):

        input_file = 'data/test_scores.h5'
        config_file = 'data/config.yaml'

        data_dict, data_metadata = load_pcs(filename=input_file,
                                            var_name='scores',
                                            npcs=10,
                                            load_groups=True)

        with open(config_file, 'r') as f:
            config_data = yaml.safe_load(f)
            config_data['hold_out'] = False

        training_data, validation_data, nt_frames = get_training_data_splits(config_data, data_dict)

        assert len(list(training_data.values())[0]) > len(list(validation_data.values())[0]), "Data split is incorrect"
        assert len(list(training_data.keys())) == len(list(validation_data.keys())), \
            "Training data and Val data do not contain same keys"

        total_frames = nt_frames[0] + len(list(validation_data.values())[0])
        percent_out = int((1 - (nt_frames[0]/total_frames)) * 100)

        assert percent_out == config_data['percent_split'], "Config file was not correctly updated"


    def test_graph_modeling_loglikelihoods(self):
        dest_file = 'data/test_model.p'
        config_file = 'data/config.yaml'

        iter_lls = [[12], [15], [19]]
        iter_holls = [[2], [5], [9]]
        group_idx = ['default', 'default', 'default']

        with open(config_file, 'r') as f:
            config_data = yaml.safe_load(f)
            config_data['hold_out'] = True
            config_data['verbose'] = True

        img_path = graph_modeling_loglikelihoods(config_data, iter_lls, iter_holls, group_idx, dest_file)

        assert os.path.exists(img_path), "Something went wrong; graph was not created."
        os.remove(img_path)