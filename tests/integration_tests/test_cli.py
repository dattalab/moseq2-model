import os
import shutil
from unittest import TestCase
from click.testing import CliRunner
from moseq2_model.cli import learn_model, count_frames, kappa_scan_fit_models

class TestCLI(TestCase):
    def test_count_frames(self):

        input_file = 'data/test_scores.h5'

        runner = CliRunner()

        result = runner.invoke(count_frames,
                               [input_file],
                               catch_exceptions=False)

        assert result.exit_code == 0, "CLI Command did not successfully complete"

    def test_learn_model(self):

        input_file = 'data/test_scores.h5'
        dest_file = 'data/test/model.p'
        checkpoint_path = 'data/test/checkpoints/'

        index = 'data/test_index.yaml'
        num_iter = 10
        freq = 2
        max_states = 100
        npcs = 10
        kappa = None

        train_params = [input_file, dest_file, "-i", index, '-n', num_iter, '--checkpoint-freq', freq,
                        '-m', max_states, '--npcs', npcs, '--e-step', '-k', kappa, '--robust', '--use-checkpoint', '--verbose']

        print(' '.join([str(s) for s in train_params]))

        runner = CliRunner()

        result = runner.invoke(learn_model,
                               train_params,
                               catch_exceptions=False)

        assert result.exit_code == 0, "CLI Command did not successfully complete"
        assert os.path.exists(dest_file), "Trained model file was not created or is in the incorrect location"
        assert os.path.exists(checkpoint_path)
        assert len(os.listdir(checkpoint_path)) == 5 # iters: 1, 3, 5, 7, 9

        assert os.path.exists('data/test/train_val0_summary.png'), 'Training logLikes summary was not created.'

        shutil.rmtree('data/test/')

    def test_kappa_scan(self):
        input_file = 'data/test_scores.h5'
        dest_dir = 'data/models/'

        index = 'data/test_index.yaml'
        num_iter = 5
        max_states = 20
        npcs = 7

        kappa_scan_params = [input_file, dest_dir, '-i', index, '-n', num_iter,
                        '-m', max_states, '--npcs', npcs, '--robust',
                        '--min-kappa', 1e4, '--max-kappa', 1e8, '--out-script', 'train_out.sh']

        runner = CliRunner()

        result = runner.invoke(kappa_scan_fit_models,
                               kappa_scan_params,
                               catch_exceptions=False)

        assert result.exit_code == 0, "CLI Command did not successfully complete"
        assert os.path.exists('./data/models/train_out.sh')
        shutil.rmtree('./data/models/')