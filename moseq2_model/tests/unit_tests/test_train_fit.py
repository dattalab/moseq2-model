import pandas as pd
from unittest import TestCase
from collections import OrderedDict
from moseq2_model.util import load_pcs
from moseq2_model.train.util import whiten_all
from moseq2_model.train.fit import _ensure_odict, _in_notebook, MoseqModel

class TestTrainFit(TestCase):

    def test_ensure_odict(self):
        input_file = 'data/test_scores.h5'

        data_dict, data_metadata = load_pcs(filename=input_file,
                                            var_name='scores',
                                            npcs=10,
                                            load_groups=True)

        assert isinstance(data_dict, OrderedDict)
        assert isinstance(_ensure_odict({'a':1, 'b':2}), OrderedDict)

    def test_model_get_params(self):
        model = MoseqModel()
        assert model.params == model.get_params()

    def test_model_set_params(self):
        model = MoseqModel()
        params = model.params
        params['silent'] = True
        params['max_states'] = 50
        model = model.set_params(**params)
        new_params = model.get_params()
        assert params['silent'] == new_params['silent']
        assert params['max_states'] == new_params['max_states']

    def test_model_fit(self):
        model = MoseqModel()
        params = model.params
        params['silent'] = True
        params['max_states'] = 50
        model = model.set_params(**params)

        input_file = 'data/test_scores.h5'

        data_dict, data_metadata = load_pcs(filename=input_file,
                                            var_name='scores',
                                            npcs=10,
                                            load_groups=True)

        model.iters = 5
        assert not hasattr(model, 'df')

        X = whiten_all(data_dict)

        model = model.fit(X)
        assert hasattr(model, 'df')

    def test_model_predict(self):
        model = MoseqModel()
        params = model.params
        params['silent'] = True
        params['max_states'] = 50
        model = model.set_params(**params)

        input_file = 'data/test_scores.h5'

        data_dict, data_metadata = load_pcs(filename=input_file,
                                            var_name='scores',
                                            npcs=10,
                                            load_groups=True)

        model.iters = 5
        assert not hasattr(model, 'df')

        X = whiten_all(data_dict)

        model = model.fit(X)

        preds = model.predict(whiten_all(data_dict))
        assert len(preds['5c72bf30-9596-4d4d-ae38-db9a7a28e912']) == 905
        assert isinstance(preds, dict)

    def test_model_ll_score(self):
        model = MoseqModel()
        params = model.params
        params['silent'] = True
        params['max_states'] = 50
        model = model.set_params(**params)

        input_file = 'data/test_scores.h5'

        data_dict, data_metadata = load_pcs(filename=input_file,
                                            var_name='scores',
                                            npcs=10,
                                            load_groups=True)

        model.iters = 5
        assert not hasattr(model, 'df')
        X = whiten_all(data_dict)

        model = model.fit(X)

        lls_dict = model.log_likelihood_score(X)
        assert len(lls_dict) == 2

        lls_list = model.log_likelihood_score(list(X.values()))
        assert len(lls_list) == 2

        lls_sum = model.log_likelihood_score(list(X.values()), reduction='sum')
        assert lls_sum == sum(lls_list)

        lls_mean = model.log_likelihood_score(list(X.values()), reduction='mean')
        assert lls_mean == sum(lls_list)/len(lls_list)


    def test_model_get_median_duration(self):
        model = MoseqModel()
        params = model.params
        params['silent'] = True
        params['max_states'] = 50
        model = model.set_params(**params)

        input_file = 'data/test_scores.h5'

        data_dict, data_metadata = load_pcs(filename=input_file,
                                            var_name='scores',
                                            npcs=10,
                                            load_groups=True)

        model.iters = 5
        assert not hasattr(model, 'df')
        X = whiten_all(data_dict)

        model = model.fit(X)

        med_dur = model.get_median_duration()
        assert isinstance(med_dur, pd.Series)
        assert len(med_dur.values) == 2


    def test_model_duration_score(self):
        model = MoseqModel()
        params = model.params
        params['silent'] = True
        params['max_states'] = 50
        model = model.set_params(**params)

        input_file = 'data/test_scores.h5'

        data_dict, data_metadata = load_pcs(filename=input_file,
                                            var_name='scores',
                                            npcs=10,
                                            load_groups=True)

        model.iters = 5
        assert not hasattr(model, 'df')
        X = whiten_all(data_dict)

        model = model.fit(X)

        dur_score = model.duration_score()
        assert isinstance(dur_score, float)
        assert dur_score != None

    def test_in_notebook(self):
        x = _in_notebook()
        assert True == x