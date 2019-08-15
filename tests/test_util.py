import pytest
import os
import numpy as np
import scipy
import joblib
import h5py
import copy
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

def test_copy_model():
    # original params: self
    mock_model_path = 'tests/test_data/mock_model.p'
    mock_model = joblib.load(mock_model_path)
    tmp = []

    # make a deep copy of the data-less version


    cp = copy.deepcopy(mock_model)

    assert(type(cp) == type(mock_model))
    assert(len(cp) == len(mock_model))
    assert(cp.keys() == mock_model.keys())
    assert all([c == d for a, b in zip(cp.values(), mock_model.values()) for c, d in zip(a, b)])

def get_params_from_model():
    # original params: model, save_ar=True
    pytest.fail('not implemented')


def test_save_dict():
    # original params: filename, obj_to_save=None

    filename = 'tests/test_data/test_dict_save'
    exts = ['.mat', '.z', '.pkl', '.p', '.h5']

    obj = {'a':1, 'b':2}
    for ext in exts:
        if (filename+ext).endswith('.mat'):
            print('Saving MAT file ' + filename)
            scipy.io.savemat(filename, mdict=obj)
            assert(os.path.exists(filename+ext))
        elif (filename+ext).endswith('.z'):
            print('Saving compressed pickle ' + filename+ext)
            joblib.dump(obj, filename+ext, compress=3)
            assert (os.path.exists(filename + ext))
        elif (filename+ext).endswith('.pkl') | (filename+ext).endswith('.p'):
            print('Saving pickle ' + filename)
            joblib.dump(obj, filename+ext, compress=0)
            assert (os.path.exists(filename + ext))
        elif (filename+ext).endswith('.h5'):
            print('Saving h5 file ' + filename+ext)
            #with h5py.File(filename, 'w') as f:
                #recursively_save_dict_contents_to_group(f, obj)
        else:
            pytest.fail(f'did not understand filetype {filename+ext}')
            raise ValueError('Did understand filetype')


def test_load_matlab_data(script_loc):

    cwd = str(script_loc)
    pcs = load_data_from_matlab(os.path.join(cwd, 'test_data/dummy_matlab.mat'),
                                var_name='features')
    keys = list(pcs.keys())

    assert(len(keys) == 1)
    assert(keys[0] == 0)
    assert(np.all(pcs[0] == 1))


def test_load_cell_string_from_matlab(script_loc):

    cwd = str(script_loc)
    groups = load_cell_string_from_matlab(os.path.join(cwd, 'test_data/dummy_matlab.mat'),
                                          var_name='groups')

    assert(len(groups) == 1)
    assert(groups[0] == 'test')


def test_load_pcs(script_loc):

    # first test loading dummy matlab file, then pickle, then h5

    cwd = str(script_loc)
    pcs, metadata = load_pcs(os.path.join(cwd,
                                          'test_data/dummy_matlab.mat'), load_groups=True)

    assert(len(pcs) == 1)
    assert(len(metadata['groups']) == 1)
    assert(np.all(pcs[0] == 1))
    assert(metadata['groups'][0] == 'test')
