from functools import partial
import math
from multiprocessing.pool import Pool, ThreadPool

import numpy as np
from skimage.segmentation import find_boundaries

from ._metrics import UnsupervisedMetric
from .distance import ecs, euclid

import warnings


class frc(UnsupervisedMetric):
    """Calculates Rosenberger (F_rc) score for a segmentation

    For each region we calculate an intra-region disparity and inter-region disparity.
    Currently only the 'uniform' calculation is implemented ('textured' would be the other).
    For uniform, the intra-region disparity is 'the normalized standard deviation'.
    This is how the paper originally calculates the intra-region disparity for a single channel.
        D_intra_r = sd_r / mean_r
    Where sd_r is the standard deviation of the region, and mean_r is the mean for the region.
    Note to get this value you will need to specify norm=True.
    The inter-region disparity is the average disparity with region r and it's neighbors,
    where the disparity between two regions r and s is:
        D(r, s) = |mean_r - mean_s| / NG
    where NG is the number of gray levels.
    Because hyperspectral images are continuous, we use NG = range(img) (that is the max pixel value minus the min pixel value)
    The average inter and intra disparity scores are taken using the region size as weights.
    The final score is defined as:
        F_rc = (D_inter - D_intra) / 2
    where D_inter is the average inter disparity and D_intra is the average intra disparity.
    This overall score can be reformulated to be the sum of region scores:
        F_rc = sum_i^m r_i/2mN (inter_r_i - intra_r_i)
    where m is the number of segments, N is the total number of pixels, r_i is the number of pixels in region i,
    and inter_r_i and intra_r_i are the inter-intra scores for region r_i.
    So the Frc score for a single region r_i is r_i/2mN (inter_r_i - intra_r_i), note that this can be negative.
    This score can be found per channel, and then aggregated according to comb_mode (default mean).
    Note that using channel_agg=True will change the dist_func to be the euclidean distance.
    To adapt this to work with hyper spectral data we redefine D_intra_r as:
        D_intra_r = sqrt(mean(d(s, r_bar)^2)) / norm(r_bar)
    where d(s, r_bar) is the dist_func between a pixel s in the region r, and r_bar, which is the average of all pixels.
    norm(r_bar) is the norm of the average spectra r_bar.
    It was found that it is better to leave norm=false (don't divide by norm(r_bar)), though leaving it true
    will match the original calculation according to the paper for a single channel, and it is a way to extend this to
    multiple channels.
    The inter region disparity is slightly changed as well, where the disparity between two regions is:
        D(r, s) = dist(r_bar, s_bar) / range(img).
    The final score calculation is the same as the single channel case.
    In theory a good segmentation will maximize this score. This score appears to favor less/larger segmentation (under segmentation).
    Originating paper at https://ieeexplore.ieee.org/stamp/stamp.jsp?arnumber=859280&tag=1
    "GENETIC FUSION : APPLICATION TO MULTI-COMPONENTS IMAGE SEGMENTATION"

        Typical usage example:
        import HSIsuqs.metrics.distance as dist
        score = frc(indianPines, seg, dist_func=dist.cos).get_score()
    """

    def __str__(self) -> str:
        return 'Frc Score'

    def __init__(self, img, seg, dist_func=ecs, uniform=True, norm=True, channel_agg=False, comb_mode='mean',
                 n_thread=8, n_proc=8, **kwargs):
        """Initializes the parameters for calculating Frc

        Note that currently uniform=True is the only calcuation option allowed (textured isn't implemented yet).

        Args:
            img (ndarray or dict): original image, or dict with key 'img' with value of 3D array.
            seg (ndarray): segmentation mask (2D array)
            dist_func (function, optional): a function to find the distance between two spectra. Defaults to ecs.
            uniform (bool, optional): Whether to use the uniform calculation. Defaults to True.
            norm (bool, optional): Where to divide scores by the norm of the average value (see description). Defaults to False.
            channel_agg (bool, optional): whether to calculate scores for each channel and aggregate. Defaults to False.
            comb_mode (str, optional): How to aggregate the channel scores (string for an np function). Defaults to 'mean'.
            n_thread (int, optional): Number of threads for calculating per channel. Defaults to 8.
            n_proc (int, optional): Number of processes for calculating each region. Defaults to 8.
        """
        super().__init__(img, seg)
        self.dist_func = dist_func
        self.uniform = True
        self.norm = norm
        self.channel_agg = channel_agg
        self.comb_mode = comb_mode
        self.n_thread = n_thread
        self.n_proc = n_proc

    def evaluate(self, img=None, seg=None):
        """Finds the Frc score for the given segmentation.

        Args:
            img (ndarray, optional): 3D image array which created the segmentation. Defaults to None.
            seg (ndarray, optional): 2D segmentation map from image. Defaults to None.

        Returns:
            float: the Frc score
        """
        img, seg = super().evaluate(img, seg)

        if self.channel_agg:
            self.dist_func = euclid
            res = []
            with ThreadPool(self.n_thread) as p:
                kwargs = {'img': img, 'seg': seg, 'dist_func': self.dist_func,
                          'uniform': self.uniform, 'norm': self.norm, 'n_proc': self.n_proc}
                res = p.map(partial(calc_frc_channel, **kwargs),
                            range(img.shape[2]))
                p.close()
                p.join()
                res = list(res)
            self.score = eval(
                f'np.{self.comb_mode}([d_all for d_all, _, __ in res])')
            self.channel_scores = res
            return self.score
        d_all, seg_aves, seg_scores = calc_frc(img, seg, self.dist_func, self.uniform,
                                               self.norm, self.n_proc)
        self.score = d_all
        self.seg_aves, self.reg_scores = seg_aves, seg_scores
        return self.score

    def get_region_score(self, lab=None, img=None, seg=None, **kwargs):
        """Calculates the Frc score for a given region.

        If lab=None, it will calculate the score for all regions.
        Summing over all region scores will reproduce the overall score.
        If img and seg are None, then the original image and segmentation is used.
        A possible kwarg to specify is seg_aves, see calc_seg_aves.

        Args:
            lab (int, optional): label for the region of interest. Defaults to None.
            img (ndarray, optional): 3D image array which created the segmentation. Defaults to None.
            seg (ndarray, optional): 2D segmentation map from image. Defaults to None.

        Returns:
            float or dict: returns score for specificed region or dict of scores for all regions.
        """
        img, seg = super().evaluate(img, seg)

        if lab is None:
            res = {}
            seg_aves = calc_seg_aves(img, seg)
            for l in np.unique(seg):
                res[l] = self.get_region_score(l, img, seg, seg_aves=seg_aves)
            return res
        seg_aves = kwargs.get('seg_aves', None)
        return calc_frc_reg(lab, img, seg, self.dist_func, self.uniform, seg_aves, self.norm)


def get_pixels(lab, img, seg):
    """Returns the pixels in a region.

    Args:
        lab (int): Label for the region
        img (ndarray, optional): 3D image array which created the segmentation
        seg (ndarray, optional): 2D segmentation map from image


    Returns:
        ndarray: 2D array of pixels (each row is a spectrum in the region)
    """
    region = np.argwhere(seg == lab)
    return np.array([img[s[0], s[1], :] for s in region])


def calc_intra_disp(region_lab, img, seg, dist_func, uniform=True, seg_aves=None, norm=False):
    """Calculates the intra region disparity

    seg_aves should come from calc_seg_aves, or be a dict with keys being all segmentation labels,
    and values being the average value or average spectrum for the region.

    Args:
        region_lab (int): label for the region to calculate score for
        img (ndarray, optional): 3D image array which created the segmentation
        seg (ndarray, optional): 2D segmentation map from image
        dist_func (function): function to calculate the distance between two spectrum
        uniform (bool, optional): Whether to calculate the uniform score. Defaults to True.
        seg_aves (dict, optional): segmentation average spectrums (from calc_seg_aves). Defaults to None.
        norm (bool, optional): Whether to divide score by the norm of the average value. Defaults to False.

    Returns:
        float: the intra region disparity
    """
    if seg_aves is None:
        seg_aves = calc_seg_aves(img, seg)
    if uniform:
        pixels = get_pixels(region_lab, img, seg)
        ave_pix = seg_aves[region_lab]['mean']
        dists = np.array([dist_func(x, ave_pix) for x in pixels])
        sd = np.sqrt(np.mean(np.square(dists)))
        if norm:
            mu = np.linalg.norm(ave_pix)
            if math.isclose(mu, 0):
                return sd
            return sd / mu
        return sd


def calc_inter_disp(region_lab, img, seg, dist_func, uniform=True, seg_aves=None, norm=False):
    """Calculates the inter region disparity score

    Args:
        region_lab (int): label for the region to calculate score for
        img (ndarray, optional): 3D image array which created the segmentation
        seg (ndarray, optional): 2D segmentation map from image
        dist_func (function): function to calculate the distance between two spectrum
        uniform (bool, optional): Whether to calculate the uniform score. Defaults to True.
        seg_aves (dict, optional): segmentation average spectrums (from calc_seg_aves). Defaults to None.
        norm (bool, optional): Whether to divide score by the range of the image. Defaults to False.

    Returns:
        float: the inter region disparity score
    """
    if seg_aves is None:
        seg_aves = calc_seg_aves(img, seg)
    if uniform:
        boundary = find_boundaries(seg == region_lab, mode='outer')
        boundary = np.argwhere(boundary)
        neighbor_labs = np.unique([seg[s[0], s[1]] for s in boundary])
        res = []
        for neigh_l in neighbor_labs:
            res.append(
                dist_func(seg_aves[region_lab]['mean'], seg_aves[neigh_l]['mean']))
        if len(res) == 0 :
            return -math.inf
        ave = np.mean(res)
        if norm:
            return ave / (np.max(img) - np.min(img))
        return ave


def calc_seg_aves(img, seg):
    """Finds the average for each region

    If it is a single channel (should still be a 3D array) then this will be the mean of the values.
    If it is multiple channels, it finds the average spectrum across all channels.
    Returns a dict with keys being the region label, and the value being the average.

    Args:
        img (ndarray, optional): 3D image (even if single channel) array which created the segmentation
        seg (ndarray, optional): 2D segmentation map from image

    Returns:
        dict: each region's average
    """
    res = {}
    for l in np.unique(seg):
        pixels = get_pixels(l, img, seg)
        res[l] = {'mean': np.mean(pixels, axis=0), 'size': pixels.shape[0]}
    return res


def calc_frc(img, seg, dist_func, uniform=True, norm=False, n_proc=8):
    """Calculates the Frc score for the segmentation

    Returns 3 values.
    The first is the overall score.
    The second is the segmentaiton averages
    the third is the inter-intra scores for each region.

    Args:
        img (ndarray, optional): 3D image array which created the segmentation
        seg (ndarray, optional): 2D segmentation map from image
        dist_func (function): function to calculate the distance between two spectrum
        uniform (bool, optional): Whether to calculate the uniform score. Defaults to True.
        norm (bool, optional): Whether to norm the inter and intra scores. Defaults to False.
        n_proc (int, optional): number of processes for the Pool. Defaults to 8.

    Returns:
        tuple: score, seg_aves, and region scores
    """
    labs = np.unique(seg)
    seg_aves = calc_seg_aves(img, seg)
    res = []
    with Pool(n_proc) as p:
        kwargs = {'img': img, 'seg': seg, 'dist_func': dist_func,
                  'uniform': uniform, 'norm': norm, 'seg_aves': seg_aves}
        res = p.map(partial(calc_frc_reg, **kwargs), labs)
        p.close()
        p.join()
        res = list(res)
    intras = []
    inters = []
    for ia, ie in res:
        intras.append(ia)
        inters.append(ie)
    inters = (inters - np.min(inters)) / (np.max(inters) - np.min(inters))
    intras = (intras - np.min(intras)) / (np.max(intras) - np.min(intras))
    m = len(seg_aves)
    N = seg.size
    r_scores = []
    for i in range(len(labs)):
        r_i = seg_aves[labs[i]]['size']
        r_scores.append((r_i / (2 * N * m)) * (inters[i] - intras[i]))
    F_rc = np.sum(r_scores)
    return F_rc, seg_aves, r_scores


def calc_frc_reg(region_lab, img, seg, dist_func, uniform, seg_aves, norm):
    """calculates the Frc score the a given region.

    Returns the score for the region.

    Args:
        region_lab (int): label for the region to calculate score for
        img (ndarray, optional): 3D image array which created the segmentation
        seg (ndarray, optional): 2D segmentation map from image
        dist_func (function): function to calculate the distance between two spectrum
        uniform (bool, optional): Whether to calculate the uniform score. Defaults to True.
        seg_aves (dict, optional): segmentation average spectrums (from calc_seg_aves). Defaults to None.
        norm (bool, optional): Whether to norm the inter and intra scores. Defaults to False.

    Returns:
        float: region score
    """
    intra = calc_intra_disp(region_lab, img, seg,
                            dist_func, uniform, seg_aves, norm)
    inter = calc_inter_disp(region_lab, img, seg,
                            dist_func, uniform, seg_aves, norm)
    # m = len(seg_aves)
    # N = seg.size
    # r_i = seg_aves[region_lab]['size']
    # r_score = (r_i / (2 * N * m)) * (inter - intra)
    return intra, inter


def calc_frc_channel(c, img, seg, dist_func, uniform=True, norm=False, n_proc=8):
    """Calculates the Frc score for a given channel.

    Wraps calc_frc where the image is just channel c.

    Args:
        c (int): label for the region to calculate score for
        img (ndarray, optional): 3D image array which created the segmentation
        seg (ndarray, optional): 2D segmentation map from image
        dist_func (function): function to calculate the distance between two spectrum
        uniform (bool, optional): Whether to calculate the uniform score. Defaults to True.
        norm (bool, optional): Whether to norm the inter and intra scores. Defaults to False.
        n_proc (int, optional): Number of processes for Pool. Defaults to 8.

    Returns:
        tuple: score, seg_aves, and region scores
    """
    img = img[:, :, c]
    img = np.atleast_2d(img).reshape((img.shape[0], img.shape[1], 1))
    return calc_frc(img, seg, dist_func, uniform, norm, n_proc)
