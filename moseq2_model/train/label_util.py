import pandas as pd
import numpy as np


def syll_onset(labels: np.ndarray) -> np.ndarray:
    '''Finds indices of syllable onsets
    Args:
        labels: array of syllable labels for a mouse
    Returns:
        an array of indices denoting the beginning of each syllables
    '''
    change = np.diff(labels) != 0
    indices = np.where(change)[0]
    indices += 1
    indices = np.concatenate(([0], indices))
    return indices


def syll_duration(labels: np.ndarray) -> np.ndarray:
    '''Computes the duration of each syllable

    >>> syll_duration(np.array([1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3, 3]))
    array([3, 4, 5])
    '''
    onsets = np.concatenate((syll_onset(labels), [labels.size]))
    durations = np.diff(onsets)
    return durations


def syll_id(labels: np.ndarray) -> np.ndarray:
    '''Returns the syllable label at each syllable transition.
    Args:
        labels: array of syllable labels for a mouse
    Returns:
        an array of compressed labels

    >>> syll_id(np.array([1, 1, 1, 2, 2, 2, 3, 3, 3]))
    array([1, 2, 3])
    '''
    onsets = syll_onset(labels)
    return labels[onsets]


def to_df(labels, uuid) -> pd.DataFrame:
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

