import pytest
import os
import numpy as np
# import numpy.testing as npt
# import uuid
from moseq2_model.util import load_data_from_matlab, load_cell_string_from_matlab,\
    load_pcs


# https://stackoverflow.com/questions/34504757/
# get-pytest-to-look-within-the-base-directory-of-the-testing-script
@pytest.fixture(scope="function")
def script_loc(request):
    return request.fspath.join('..')


@pytest.fixture(scope='function')
def temp_dir(tmpdir):
    f = tmpdir.mkdir('test_dir')
    return str(f)


def test_load_matlab_data(script_loc):

    pcs = load_data_from_matlab('data/dummy_matlab.mat', var_name='features')
    keys = list(pcs.keys())

    assert(len(keys) == 1)
    assert(keys[0] == 0)
    assert(np.all(pcs[0] == 1))


def test_load_cell_string_from_matlab(script_loc):

    groups = load_cell_string_from_matlab('data/dummy_matlab.mat', var_name='groups')

    assert(len(groups) == 1)
    assert(groups[0] == 'test')


def test_load_pcs(script_loc):

    # first test loading dummy matlab file, then pickle, then h5
    pcs, metadata = load_pcs('data/dummy_matlab.mat', load_groups=True)

    assert(len(pcs) == 1)
    assert(len(metadata['groups']) == 1)
    assert(np.all(pcs[0] == 1))
    assert(metadata['groups'][0] == 'test')
