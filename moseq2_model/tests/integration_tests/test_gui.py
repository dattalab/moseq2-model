import os
import sys
import shutil
import joblib
from unittest import TestCase
from moseq2_model.gui import learn_model_command
from tempfile import TemporaryDirectory, NamedTemporaryFile

class TestGUI(TestCase):

    def test_learn_model(self):

        input_file = 'data/test_scores.h5'
        dest_file = 'data/model.p'
        config_file = 'data/config.yaml'
        index = 'data/test_index.yaml'
        checkpoint_path = 'data/checkpoints/'

        hold_out = True
        nfolds = 2
        num_iter = 10
        max_states = 100
        npcs = 10
        kappa = None
        separate_trans = True
        robust = True
        checkpoint_freq = 2
        percent_split=20
        verbose = False

        with TemporaryDirectory() as tmp:
            # test space-separated input
            stdin = NamedTemporaryFile(prefix=tmp+'/', suffix=".txt")
            with open(stdin.name, 'w') as f:
                f.write('default Group1')
            f.close()

            sys.stdin = open(stdin.name)

        learn_model_command(input_file, dest_file, config_file, index, hold_out, nfolds, num_iter,
                max_states, npcs, kappa, separate_trans, robust, checkpoint_freq, percent_split, verbose)

        assert (os.path.exists(dest_file)), "Trained model file was not created or is in the incorrect location"
        assert (os.path.exists(checkpoint_path)), "Checkpoints were not created"
        assert len(os.listdir(checkpoint_path)) == 5  # iters: 1, 3, 5, 7, 9

        num_iter = 15 # train for 5 more iterations
        updated_dest_file = 'data/updated_model.p'

        with TemporaryDirectory() as tmp:
            # test space-separated input
            stdin = NamedTemporaryFile(prefix=tmp+'/', suffix=".txt")
            with open(stdin.name, 'w') as f:
                f.write('default Group1')
            f.close()

            sys.stdin = open(stdin.name)


        learn_model_command(input_file, updated_dest_file, config_file, index, hold_out, nfolds, num_iter,
                            max_states, npcs, kappa, separate_trans, robust, checkpoint_freq, percent_split, verbose)

        assert (os.path.exists(updated_dest_file)), "Updated model file was not created or is in the incorrect location"
        shutil.rmtree(checkpoint_path)

        model1 = joblib.load('data/model.p')
        model2 = joblib.load('data/updated_model.p')

        assert model1 != model2

        os.remove(dest_file)
        os.remove(updated_dest_file)
