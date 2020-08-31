'''
Syllable label information utility functions.
'''

import pandas as pd
import numpy as np


def syll_onset(labels: np.ndarray) -> np.ndarray:
    '''
    Finds indices of syllable onsets.

    Parameters
    ----------
    labels (np.ndarray): array of syllable labels for a mouse.

    Returns
    -------
    indices (np.ndarray): an array of indices denoting the beginning of each syllables.
    '''

    # Getting indices of syllable switching
    change = np.diff(labels) != 0
    indices = np.where(change)[0]

    # Getting frame indices of the switched syllable
    indices += 1
    indices = np.concatenate(([0], indices))
    return indices


def syll_duration(labels: np.ndarray) -> np.ndarray:
    '''
    Computes the duration of each syllable.

    Parameters
    ----------
    labels (np.ndarray): array of syllable labels for a mouse.

    >>> syll_duration(np.array([1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3, 3]))
    array([3, 4, 5])

    Returns
    -------
    durations (np.ndarray): array of syllable durations.
    '''

    onsets = np.concatenate((syll_onset(labels), [labels.size]))
    durations = np.diff(onsets)
    return durations


def syll_id(labels: np.ndarray) -> np.ndarray:
    '''
    Returns the syllable label at each syllable transition.

    Parameters
    ----------
    labels (np.ndarray): array of syllable labels for a mouse.

     >>> syll_id(np.array([1, 1, 1, 2, 2, 2, 3, 3, 3]))
     array([1, 2, 3])

    Returns
    -------
    labels[onsets] (np.ndarray): an array of compressed labels.
    '''

    onsets = syll_onset(labels)
    return labels[onsets]


def to_df(labels, uuid) -> pd.DataFrame:
    '''
    Convert labels numpy.ndarray to pandas.DataFrame

    Parameters
    ----------
    labels (np.ndarray): array of syllable labels for a mouse.
    uuid (list): list of session uuids representing each series of labels.

    Returns
    -------
    df (pd.DataFrame): DataFrame of syllables, durations, onsets, and session uuids.
    '''

    if isinstance(labels, pd.Series):
        labels = labels.to_numpy()
    elif isinstance(labels, list):
        labels = np.array(labels, dtype=np.int32)

    # Packing all syllable information into a single DataFrame
    df = pd.DataFrame({
        'syll': syll_id(labels),
        'dur': syll_duration(labels),
        'onset': syll_onset(labels),
        'uuid': uuid
    })

    return df