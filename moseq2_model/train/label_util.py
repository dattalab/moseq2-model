import pandas as pd
import numpy as np


def syll_onset(labels):
    change = np.diff(labels) != 0
    indices = np.where(change)[0]
    indices += 1
    indices = np.concatenate(([0], indices))
    return indices


def syll_duration(labels):
    onsets = np.concatenate((syll_onset(labels), [labels.size]))
    durations = np.diff(onsets)
    return durations


def syll_id(labels):
    onsets = syll_onset(labels)
    return labels[onsets]


def to_df(labels, uuid):
    if isinstance(labels, pd.Series):
        labels = labels.to_numpy()
    elif isinstance(labels, list):
        labels = np.array(labels, dtype=np.int32)

    df = pd.DataFrame({
        'syll': syll_id(labels),
        'dur': syll_duration(labels),
        'onset': syll_onset(labels),
        'uuid': uuid
    })

    return df

