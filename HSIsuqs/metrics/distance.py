from math import acos, log

import numpy as np
from scipy.spatial.distance import cosine, euclidean
from scipy.special import rel_entr


def ecs(s1, s2, sym=True, norm=True):
    """Calculates the Euclidean distance of cumulative spectrum.

    Distance function which can be found at https://tel.archives-ouvertes.fr/tel-01471226/file/these_HildaDeborah_FR_fixed.pdf
    "Towards Spectral Mathematical Morphology"
    This method implicitly puts more weight on earlier spectrum, so to symmetrize it,
    it will take the average of the original calculation, and the calculation with the spectrum flipped (reversed).
    Normalizing will just divide the spectrum by it's sum (to turn it into a pseudo CDF).

    Args:
        s1 (ndarray): 1D spectrum
        s2 (ndarray): 1D spectrum
        sym (bool, optional): To calculate a symmetric version. Defaults to False.
        norm (bool, optional): To normalize the spectrum first (divide by sum). Defaults to True.

    Returns:
        float: the ECS distance between s1 and s2
    """
    if norm and len(s1) > 1:
        s1, s2 = s1 / np.sum(s1), s2 / np.sum(s2)
    if sym is True:
        first = ecs(s1, s2, norm=False, sym=False)
        second = ecs(np.flip(s1), np.flip(s2), norm=False, sym=False)
        return (first + second) / 2

    s1_integ = np.cumsum(s1)
    s2_integ = np.cumsum(s2)
    integ = np.sum(np.square(s1_integ - s2_integ))
    return np.math.sqrt(integ)


def _kl_div(s1, s2, eps=10e-32):
    """Calculates KL divergence (sort of)

    Calculates KL divergence for all values of s1 AND s2 that are non-zero.
    This way it won't return inf.

    Args:
        s1 (ndarray): array of probabilities
        s2 (ndarray): array of probabilities

    Returns:
        float: KL(s1 || s2)
    """
    # w = np.logical_and(s1 != 0, s2 != 0)
    # s1, s2 = s1[w], s2[w]
    # d = np.divide(s1, s2)
    # return np.sum(s1 * np.log(d))
    w = np.logical_or(s1 == 0, s2 == 0)
    s1[w] += eps
    s2[w] += eps
    return np.sum(rel_entr(s1, s2))


def klpd(s1, s2, norm=False, eps=10e-32):
    """Calculates the Spectral Kullback-Leibler Pseudo-Divergence

    Distance function based on KL divergence, found at https://tel.archives-ouvertes.fr/tel-01471226/file/these_HildaDeborah_FR_fixed.pdf
    "Towards Spectral Mathematical Morphology"

    Args:
        s1 (ndarray): 1D spectrum
        s2 (ndarray): 1D spectrum
        norm (bool, optional): whether to normalize the spectrum first. Defaults to True.

    Returns:
        float: The score as described above.
    """
    if norm and len(s1) > 1:
        s1, s2 = s1 / np.sum(s1), s2 / np.sum(s2)
    k1 = np.sum(s1)
    k2 = np.sum(s2)
    s1_ = s1 / k1
    s2_ = s2 / k2
    first = k1 * _kl_div(s1_, s2_, eps)
    second = k2 * _kl_div(s2_, s1_, eps)
    third = (k1 - k2) * (log(k1) - log(k2))
    return first + second + third


def cos(s1, s2, norm=False):
    """Calculates the cosine similarity between spectra

    Args:
        s1 (ndarray): 1D spectrum array
        s2 (ndarray): 1D spectrum array
        norm (bool, optional): Whether to normalize the spectrum first. Defaults to True.

    Returns:
        float: the score
    """
    if norm and len(s1) > 1:
        s1, s2 = s1 / np.sum(s1), s2 / np.sum(s2)
    return cosine(s1, s2)


def sam(s1, s2, norm=False):
    """Calculates the Spectral Angle Mappers value.

    Args:
        s1 (ndarray): 1D spectrum array
        s2 (ndarray): 1D spectrum array
        norm (bool, optional): Whether to normalize the spectrum first. Defaults to True.

    Returns:
        float: the score
    """
    if norm and len(s1) > 1:
        s1, s2 = s1 / np.sum(s1), s2 / np.sum(s2)
    sim = 1 - cos(s1, s2)
    return acos(sim)


def euclid(s1, s2, norm=False):
    """Calculates the euclidean distance between spectrums

    Args:
        s1 (ndarray): 1D spectrum array
        s2 (ndarray): 1D spectrum array
        norm (bool, optional): Whether to normalize the spectrum first. Defaults to True.

    Returns:
        float: the score
    """
    if norm and len(s1) > 1:
        s1, s2 = s1 / np.sum(s1), s2 / np.sum(s2)
    return euclidean(s1, s2)
