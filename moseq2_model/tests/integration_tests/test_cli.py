import os
from unittest import TestCase
from click.testing import CliRunner
from moseq2_model.cli import learn_model, count_frames

class TestCLI(TestCase):
    def test_count_frames(self):

        input_file = 'data/test_scores.h5'

        runner = CliRunner()

        result = runner.invoke(count_frames,
                               [input_file],
                               catch_exceptions=False)

        assert result.exit_code == 0

    def test_learn_model(self):

        input_file = 'data/test_scores.h5'
        dest_file = 'data/model.p'


        index = 'data/test_index.yaml'
        num_iter = 10
        max_states = 100
        npcs = 10
        kappa = None

        train_params = [input_file, dest_file, "-i", index, '-n', num_iter,
                        '-m', max_states, '--npcs', npcs, '-k', kappa, '--robust']

        runner = CliRunner()

        result = runner.invoke(learn_model,
                               train_params,
                               catch_exceptions=False)

        assert os.path.exists(dest_file)
        assert result.exit_code == 0
        os.remove(dest_file)