import os
import sys
from unittest import TestCase
from moseq2_model.gui import learn_model_command
from tempfile import TemporaryDirectory, NamedTemporaryFile

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

        with TemporaryDirectory() as tmp:
            # test space-separated input
            stdin = NamedTemporaryFile(prefix=tmp, suffix=".txt")
            with open(stdin.name, 'w') as f:
                f.write('default Group1')
            f.close()

            sys.stdin = open(stdin.name)

        learn_model_command(input_file, dest_file, config_file, index, hold_out, nfolds, num_iter,
                max_states, npcs, kappa, separate_trans, robust, checkpoint_freq, percent_split, verbose)

        assert (os.path.exists(dest_file)), "Trained model file was not created or is in the incorrect location"
        os.remove(dest_file)