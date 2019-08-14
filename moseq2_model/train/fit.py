'''
Contains a model class that is compatible with scikit-learn's GridsearchCV api.
This class extends other functionality, such as visually inspecting model
statistics within a jupyter notebook
'''
import sys

def _in_notebook():
    '''determine if this function was executed in a jupyter notebook
    
    Returns:
        a boolean describing the presence of a jupyter notebook'''
    return 'ipykernel' in sys.modules


class MoseqModel:
    def __init__(self):
        pass

    def get_params(self):
        raise NotImplementedError()

    def set_params(self):
        raise NotImplementedError()
    
    def fit(self):
        raise NotImplementedError()

    def partial_fit(self):
        raise NotImplementedError()

    def predict(self):
        raise NotImplementedError()

    def predict_proba(self):
        raise NotImplementedError()

    def score(self):
        raise NotImplementedError()