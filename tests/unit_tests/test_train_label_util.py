import numpy as np
import pandas as pd
from unittest import TestCase
from moseq2_model.train.label_util import syll_onset, syll_duration, syll_id, to_df

class TestTrainLabelUtils(TestCase):

    def test_syll_onset(self):

        labels = [0, 0, 0, 1, 1, 1, 1, 1, 2, 2, 2, 3, 3, 3, 0, 0, 0, 0]
        change = np.diff(labels) != 0
        indices = np.where(change)[0]
        indices += 1
        indices = np.concatenate(([0], indices))
        assert all(indices == [0, 3, 8, 11, 14])
        assert all(indices == syll_onset(labels))


    def test_syll_duration(self):
        labels = [0, 0, 0, 1, 1, 1, 1, 1, 2, 2, 2, 3, 3, 3, 0, 0, 0, 0]
        change = np.diff(labels) != 0
        indices = np.where(change)[0]
        indices += 1
        indices = np.concatenate(([0], indices))

        onsets = np.concatenate((indices, [np.asarray(labels).size]))
        durations = np.diff(onsets)
        assert all(durations == [3, 5, 3, 3, 4])
        assert all(durations == syll_duration(np.asarray(labels)))

    def test_syll_id(self):
        labels = [0, 0, 0, 1, 1, 1, 1, 1, 2, 2, 2, 3, 3, 3, 0, 0, 0, 0]
        output = [0, 1, 2, 3, 0]

        assert all(syll_id(np.asarray(labels)) == output)

    def test_to_df(self):
        labels = [0, 0, 0, 1, 1, 1, 1, 1, 2, 2, 2, 3, 3, 3, 0, 0, 0, 0]
        pd_labels = pd.Series(labels)

        out1 = to_df(labels, ('id1'))
        out2 = to_df(pd_labels, ('id1'))

        assert (isinstance(out1, pd.DataFrame))
        assert (isinstance(out2, pd.DataFrame))