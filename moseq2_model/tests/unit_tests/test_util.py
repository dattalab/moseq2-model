import numpy as np
from unittest import TestCase
from moseq2_model.util import load_data_from_matlab, load_cell_string_from_matlab, load_pcs

class TestUtils(TestCase):

    def test_load_pcs(self):

        # first test loading dummy matlab file, then pickle, then h5

        pcs, metadata = load_pcs('data/dummy_matlab.mat', load_groups=True)

        assert(len(pcs) == 1)
        assert(len(metadata['groups']) == 1)
        assert(np.all(pcs[0] == 1))
        assert(metadata['groups'][0] == 'test')


    def test_save_dict(self):
        print()

    def test_recursively_save_dict_contents_to_group(self):
        print()

    def test_load_arhmm_checkpoint(self):
        print()

    def test_save_arhmm_checkpoint(self):
        print()

    def test_append_resample(self):
        print()

    def test_load_dict_from_hdf5(self):
        print()

    def test_load_h5_to_dict(self):
        print()

    def test_h5_to_dict(self):
        print()

    def test_copy_model(self):
        print()

    def test_get_parameters_from_model(self):
        print()

    def test_progressbar(self):
        print()

    def test_list_rank(self):
        print()

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
