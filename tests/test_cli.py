import pytest
import os
import numpy as np
import h5py
import ruamel.yaml as yaml
from click.testing import CliRunner
from moseq2_model.cli import *

def test_count_frames():

    input_dir = 'tests/test_data/'

    file_mat = 'dummy_matlab.mat'
    file_p = 'test_dict_save.p'
    file_z = 'test_dict_save.z'
    file_scores = 'testh5.h5'

    runner = CliRunner()
    result = runner.invoke(count_frames, ['--var-name', 'scores',
                                           os.path.join(input_dir, file_mat)])
    print(result.output)
    assert(result.exit_code == 0)

def test_learn_model():

    input_dir = 'tests/test_data/'

    runner = CliRunner()
    learn_params = ['-h', # FLAG
                    '--hold-out-seed', -1,
                    '--nfolds', 5,
                    '-c', 2,
                    '-n', 100,
                    '-r', 1,
                    '--var-name', 'scores',
                    '-s', -1,
                    '--save-model', # FLAG
                    '-m', 100,
                    '-p', True,
                    '--npcs', 10,
                    '-w', 'all',
                    '-k', None,
                    '-g', 1e3,
                    '--noise-level', 0,
                    '--nu', 4,
                    '--nlags', 3,
                    #'--separate-trans', # FLAG
                    #'--robust', # FLAG
                    #'-i', 'moseq-index.yaml',
                    #'--default-group', 'n/a',
                    os.path.join(input_dir, 'pca_scores.h5'),
                    os.path.join(input_dir, 'test_trained_model.p')]

    result = runner.invoke(learn_model, learn_params)

    assert(os.path.exists(os.path.join(input_dir, 'test_trained_model.p') == True))
    os.remove(os.path.join(input_dir, 'test_trained_model.p'))
    assert(result.exit_code == 0)