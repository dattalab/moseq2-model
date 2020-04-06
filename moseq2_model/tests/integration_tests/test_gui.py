import os
from unittest import TestCase
from moseq2_model.gui import learn_model_command

class TestGUI(TestCase):

    def test_learn_model(self):

        input_file = 'data/test_scores.h5'
        dest_file = 'data/model.p'
        config_file = 'data/config.yaml'
        index = 'data/test_index.yaml'
        hold_out = False
        nfolds = 2
        num_iter = 10
        max_states = 100
        npcs = 10
        kappa = None
        separate_trans = False
        robust = True
        checkpoint_freq = -1
        percent_split=20
        verbose = False

        learn_model_command(input_file, dest_file, config_file, index, hold_out, nfolds, num_iter,
                max_states, npcs, kappa, separate_trans, robust, checkpoint_freq, percent_split, verbose)

        assert (os.path.exists(dest_file))