'''
Contains a model class that is compatible with scikit-learn's GridsearchCV api.
This class extends other functionality, such as visually inspecting model
statistics within a jupyter notebook
'''
import sys
import numpy as np
import pandas as pd
from .models import ARHMM
from .util import get_labels_from_model
from .label_util import to_df
from copy import deepcopy
from cytoolz import merge, valmap
from collections import OrderedDict
from ipywidgets import Label
from IPython.display import display
from tqdm import tqdm, tqdm_notebook


def _ensure_odict(data):
    '''
    Casts input data to OrderedDict if it is not one already.

    Parameters
    ----------
    data (list or dict): data dictionary to train ARHMM.

    Returns
    -------
    data (OrderedDict): Ordered version of input data variable
    '''

    if isinstance(data, (list, tuple, np.ndarray)):
        data = OrderedDict(enumerate(data))
    elif isinstance(data, dict):
        data = OrderedDict(data)
    return data


class MoseqModel:
    def __init__(self, max_iters=100, n_cpus=1, optimal_duration=0.4,
                 scale_kappa_w_alpha=True, history=True, **model_params):
        '''
        Args:
            max_iters: number of iterations to train AR-HMM
            n_cpus: number of cpus to parallelize AR-HMM sampling
            optimal_duration: ideal syllable duration in seconds
            scale_kappa_w_alpha: if alpha changes, also scale kappa
        '''
        # define a set of default model parameters
        self.default_params = {
            'silent': False,
            'kappa': 1e6,
            'alpha': 5.7,
            'gamma': 1000,
            'nlags': 3,
            'max_states': 100,
            'robust': False,
            'groups': None,
            'separate_trans': False
        }

        self.cpus = n_cpus
        self.iters = max_iters
        self.scale_kappa = scale_kappa_w_alpha
        self.optimal_duration = optimal_duration * 30
        self.params = merge(self.default_params, model_params)
        self.history = history
        self.rho = self.params['kappa'] / (self.params['kappa'] + self.params['alpha'])

        if history:
            self.dur_history = []
            self.label_history = []
            self.ll_history = []

    def get_params(self, deep=True):
        '''
        Get model parameters.

        Parameters
        ----------
        deep (bool): indicate whether to use deep copy

        Returns
        -------
        params (dict): Model parameters
        '''

        if deep:
            return deepcopy(self.params)
        return self.params

    def set_params(self, **model_params):
        '''
        Update model parameters.

        Parameters
        ----------
        model_params (dict): model parameter dictionary to update

        Returns
        -------
        None
        '''

        if self.scale_kappa and 'alpha' in model_params:
            new_kappa = (model_params['alpha'] / (1 - self.rho)) - model_params['alpha']
            model_params['kappa'] = new_kappa
        self.params = merge(self.params, model_params)
        return self

    def fit(self, X, y=None):
        '''
        Trains model given data.

        Parameters
        ----------
        X (OrderedDict): data_dict used to train ARHMM
        y (None)

        Returns
        -------
        None
        '''

        X = _ensure_odict(X)
        in_nb = _in_notebook()
        silence = self.params['silent']
        if self.history:
            self.dur_history = []
            self.label_history = []
            self.ll_history = []

        arhmm = ARHMM(data_dict=deepcopy(_ensure_odict(X)), **self.params)

        progressbar = tqdm_notebook if in_nb else tqdm

        if not silence and in_nb:
            lbl = Label('duration: ll:')
            display(lbl)

        for _ in progressbar(range(self.iters), disable=silence):
            arhmm.resample_model(num_procs=self.cpus)
            labels = get_labels_from_model(arhmm)
            self.df = pd.concat([to_df(l, u) for u, l in enumerate(labels)])

            if self.history:
                _dur = self.get_median_duration().mean() / 30
                self.label_history.append(self.df.copy())
                self.ll_history.append(arhmm.log_likelihood())
                self.dur_history.append(_dur)

            if not silence and in_nb and self.history:
                lbl.value = f'median duration: {self.dur_history[-1]:0.3f}s -- log-likelihood: {self.ll_history[-1]:0.3E}'

        self.arhmm = arhmm
        return self

    def partial_fit(self, X):
        '''
        Not implemented.

        Parameters
        ----------
        X (OrderedDict)

        Returns
        -------
        '''

        X = _ensure_odict(X)
        self.arhmm = ARHMM(X, **self.params)
        raise NotImplementedError()

    def predict(self, X):
        '''
        Get label predictions from input data.

        Parameters
        ----------
        X (list, or OrderedDict): data to predict labels

        Returns
        -------
        y_pred (list): list of label predictions
        '''

        if isinstance(X, (list, tuple)):
            y_pred = [self.arhmm.heldout_viterbi(_x) for _x in X]
        elif isinstance(X, (dict, OrderedDict)):
            y_pred = valmap(self.arhmm.heldout_viterbi, X)
        else:
            raise TypeError('Data type not understood - only : (list, tuple, dict, OrderedDict)')
        return y_pred

    def predict_proba(self):
        raise NotImplementedError()

    def log_likelihood_score(self, X, reduction=None):
        '''
        Compute Log-Likelihood Score of each session.

        Parameters
        ----------
        X (list or OrderedDict): data to compute log-likelihood score from.
        reduction (str): indicates whether to use a reduction operation.

        Returns
        -------
        _lls (1D numpy array): log-likelihood arrays.
        '''

        if isinstance(X, (list, tuple)):
            _lls = map(self.arhmm.log_likelihood, X)
            _lls = [v / l for v, l in zip(_lls, map(len, X))]
        elif isinstance(X, (dict, OrderedDict)):
            _lls = valmap(self.arhmm.log_likelihood, X)
            _lls = [v / l for v, l in zip(_lls.values(), map(len, X.values()))]
        if reduction == 'sum':
            _lls = np.sum(_lls)
        elif reduction == 'mean':
            _lls = np.mean(_lls)

        return _lls

    def get_median_duration(self):
        '''
        Calculates median syllable durations for each session included in the model.

        Returns
        -------
        (pandas DataFrame): DataFrame of median syllable durations
        '''

        return self.df.groupby('uuid').median().dur

    def duration_score(self):
        '''
        Computes score for assigned syllable duration
        This score is is typically used to find the models with syllable durations
         close to the data's changepoint durations.


        Returns
        -------
        (float): a single negative number that should be maximized (to get close to 0)
        '''

        dur = self.get_median_duration().mean()
        return -np.abs(dur - self.optimal_duration)

    def score(self):
        raise NotImplementedError()