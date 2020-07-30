import os
import sys
import shutil
import joblib
from unittest import TestCase
from moseq2_model.gui import learn_model_command

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
        percent_split = 20
        verbose = True

        # test space-separated input
        stdin = 'data/stdin.txt'
        with open(stdin, 'w') as f:
            f.write('default Group1')

        sys.stdin = open(stdin)

        learn_model_command(input_file, dest_file, config_file, index, hold_out=hold_out, nfolds=nfolds,
                            num_iter=num_iter,
                            max_states=max_states, npcs=npcs, kappa=kappa, separate_trans=separate_trans, robust=robust,
                            checkpoint_freq=checkpoint_freq, percent_split=percent_split, verbose=verbose,
                            select_groups=True)

        assert (os.path.exists(dest_file)), "Trained model file was not created or is in the incorrect location"
        assert (os.path.exists(checkpoint_path)), "Checkpoints were not created"
        assert len(os.listdir(checkpoint_path)) == 5  # iters: 1, 3, 5, 7, 9

        assert os.path.exists('data/train_heldout_summary.png')
        os.remove('data/train_heldout_summary.png')

        os.rename(dest_file, 'data/original_model.p')

        num_iter = 15 # train for 5 more iterations
        checkpoint_freq = -1

        # test space-separated input
        with open(stdin, 'w') as f:
            f.write('default Group1')

        sys.stdin = open(stdin)

        learn_model_command(input_file, dest_file, config_file, index, hold_out=hold_out, nfolds=nfolds,
                            num_iter=num_iter, use_checkpoint=True,
                            max_states=max_states, npcs=npcs, kappa=kappa, separate_trans=separate_trans, robust=robust,
                            checkpoint_freq=checkpoint_freq, percent_split=percent_split, verbose=verbose,
                            select_groups=True)

        assert (os.path.exists(dest_file)), "Updated model file was not created or is in the incorrect location"
        shutil.rmtree(checkpoint_path)

        assert os.path.exists('data/train_heldout_summary.png')
        os.remove('data/train_heldout_summary.png')

        model1 = joblib.load('data/original_model.p')
        model2 = joblib.load(dest_file)

        assert model1 != model2

        os.remove('data/original_model.p')
        os.remove(dest_file)
        os.remove(stdin)
