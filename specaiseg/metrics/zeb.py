from functools import partial
from math import inf
from multiprocessing.pool import Pool, ThreadPool

import numpy as np
from scipy.spatial.distance import pdist
from skimage import segmentation

from ._metrics import UnsupervisedMetric
from .distance import ecs


class zeb(UnsupervisedMetric):
    """Calculates Zeboudj's contrast score for a segmentation.

    For each region we calculate the inside contrast (I_i) and outside constrast (E_i) where:
        I_i = 1/A_i sum_s max{c(s,t), t in neighborhood(s) inside R_i}
        E_i = 1/l_i sum_s max{c(s,t), t in neighborhood(s) outside R_i}
    Where I_i goes over all pixels inside the region R_i, and E_i goes over all pixels on the boarder of R_i.
    For contrast, if using all channels at once, then c(s,t) is just the dist_func(s, t);
    if averaging over channels, then it uses the euclidean distance (aboslute difference) the single value (s,t).
    You can also specify method='mean' or 'median' or 'rob', which will return the mean/median contrast.
    Using method='rob' is like rcmg for Watershed, it removes the r pairs of pixels with the largest contrast before returning the max.
    The regions contrast is then:
        C(R_i) = 1-I_i/E_i (for 0 < I_i < E_i), or = E_i (for I_i=0), or = 0 (otherwise)
    Though if you set cap_min=False, then C(R_i)=1-I_i/E_i.
    The final score is then:
        C_all = 1/A sum_i A_i C(R_i)
    where A is the total number of pixels, and A_i is the number of pixels in region R_i.
    This implementation uses multiprocessing.pool, so if you want to wrap this in another multiprocessing,
    the outer multiprocessing should be something like ThreadPool.
    Maximizing this score indicates a better segmentation. This score appears to favor smaller/more segments (over segmentation).
    Originating paper at https://ieeexplore.ieee.org/stamp/stamp.jsp?arnumber=1334206,
    "Unsupervised Evaluation of Image Segmentation Application to Multi-spectral Images"

        Typical usage example:
        import HSIsuqs.metrics.distance as dist
        score = zeb(indianPines['img'], seg, dist_func=dist.kl).get_score()
    """

    def __str__(self):
        return 'Zeb Score'

    def __init__(self, img, seg, channel_agg=False, dist_func=ecs,
                 n_thread=8, n_proc=8, comb_mode='mean', cap_min=True,
                 method='max', r=1, **kwargs):
        """Initializes parameters used to calculate Zeb score.

        Sets some of the parameters used to calculate the zeb score, such as whether or not to aggregate over channel,
        and what distance function to use. If channel_agg is True, then the dist_func used will be overwritten to be 'euclidean'.
        When using channel_agg, a ThreadPool with n_thread threads is opened to process the channels.
        n_proc is to control how many processes are in the Pool when calculating the Zeb score.
        So if you are not using channel_agg, then only n_proc matters for computation speed tuning.

        Args:
            img (ndarray or dict): original image, or dict with key 'img' with value of 3D array.
            seg (ndarray): segmentation mask (2D array)
            channel_agg (bool, optional): if True, will aggregate results by channel using the specified comb_mode. Defaults to False.
            dist_func (str or function, optional): a distance function which takes in two spectra (defines contrast).
                Can also be a string (equivalent to the metric argument in scipy pdist function). Defaults to ecs.
            n_thread (int, optional): the number of threads to spawn in threadpool. Defaults to 8.
            n_proc (int, optional): the number of processes to spawn in pool. Defaults to 8
            comb_mode (str, optional): the string of a numpy combination function (such as mean, min, max). Defaults to 'mean'.
            cap_min (bool, optional): if true, will make the minimum possible score be 0 (the way the original paper works). Defaults to True.
            method (str, optional): numpy function or 'rob'. Defaults to 'mean'.
            r (int, optional): if method is 'rob', then returns the rth maximum distance. Defaults to 1.

        """
        super().__init__(img, seg)
        self.channel_agg = channel_agg
        self.dist_func = dist_func
        self.n_thread = n_thread
        self.n_proc = n_proc
        self.comb_mode = comb_mode
        self.cap_min = cap_min
        self.method = method
        self.r = r

    def evaluate(self, img=None, seg=None):
        """Calculates the zeb score.

        For the given image (img) and segmentation (seg), it calculates the zeb score
        according to the hyperparameters that were initialized.

        Args:
            img (ndarray, optional): 3D image array which created the segmentation. Defaults to None.
            seg (ndarray, optional): 2D segmentation map from image. Defaults to None.

        Raises:
            ValueError: If you don't pass in img and seg, it will use the original img and seg.
            If you just pass in one and not the other, it will throw an error, you need both.

        Returns:
            float: the final score.
        """
        img, seg = super().evaluate(img, seg)

        if self.channel_agg:
            self.dist_func = 'euclidean'
            res = []
            with ThreadPool(self.n_thread) as p:
                kwargs = {'img': img,
                          'seg': seg,
                          'dist_func': self.dist_func,
                          'n_proc': self.n_proc,
                          'cap_min': self.cap_min,
                          'method': self.method,
                          'r': self.r}
                res = p.map(partial(calc_zeb_channel, **kwargs),
                            range(img.shape[2]))
                p.close()
                p.join()
                res = list(res)
            self.score = eval(
                f'np.{self.comb_mode}(res)')
            self.channel_scores = res
            return self.score

        C_all = calc_zeb(
            img, seg, dist_func=self.dist_func, n_proc=self.n_proc, cap_min=self.cap_min,
            method=self.method, r=self.r)
        self.score = C_all
        return C_all

    def get_region_score(self, lab=None, img=None, seg=None):
        """Calculates the Frc score for a given region.

        If lab=None, it will calculate the score for all regions.
        Summing over all region scores will reproduce the overall score.
        If img and seg are None, then the original image and segmentation is used.

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
            for l in np.unique(seg):
                res[l] = self.get_region_score(l, img, seg)
            return res
        return calc_zeb_reg(lab, img, seg, self.dist_func, self.cap_min, self.method, self.r)


def get_cond_neighbor(r, c, img, cond, size=1):
    """Get neighbors of pixel with a given condition.

    For a pixel at (r,c) in img, with a given condition matrix (like a mask),
    it returns all the neighboring pixels (window of 1) where the conditions are true.

    Args:
        r (int): the row of the pixel of interest
        c (int): the column of the pixel of interest
        img (ndarray): the original image which was segmented
        cond (ndarray): array with True/False condition, same shape as img

    Returns:
        ndarray: returns a list of the neighbors of pixel (r, c) from the image.
    """
    width, height = img.shape[1], img.shape[0]
    ri, ui = max(c - size, 0), max(r - size, 0)
    di, li = min(r + size + 1, height), min(c + size + 1, width)
    temp = np.full((height, width), False)
    temp[ui:di, ri:li] = True
    final_cond = np.logical_and(temp, cond)
    xs, ys = np.where(final_cond)
    if img.ndim > 2:
        return img[xs, ys, :]
    return img[xs, ys]


def get_contrast(pixels, dist_func, method='max', r=1):
    """Given a set of pixels, returns the max contrast.

    For a set of pixels, this uses the dist_func to find the pairwise distances.
    This is how we define max contrast for a pixel.
    The method can be a numpy funciton, such as max, mean, median.
    Or method can be 'rob' wich will return the rth max distance (just like rcmg for Watershed).
    r=1 means remove the first pair of pixels with the largest distance. r=0 would just return the maximum distance.

    Args:
        pixels (ndarray): array of pixels (from _get_cond_neighbor)
        dist_func (function): function to define contrast (or distance) between pixels
        method (str, optional): numpy function or 'rob'. Defaults to 'mean'.
        r (int, optional): if method is 'rob', then returns the rth maximum distance. Defaults to 1.

    Returns:
        float: the max value of dist_func for each pair of pixels
    """
    n_coor = len(pixels)
    dists = np.zeros((n_coor, n_coor))
    row, col = np.triu_indices(n_coor, 1)
    if pixels.ndim < 2:
        pixels = np.atleast_2d(pixels).reshape(pixels.shape[0], 1)
    dists[row, col] = pdist(pixels, dist_func)
    if method == 'rob':
        if dists.shape[0] <= 2:
            return dists.max()
        for i in range(r):
            row, col = np.where(dists == dists.max())
            dists = np.delete(dists, [row[0], col[0]], 0)
            dists = np.delete(dists, [row[0], col[0]], 1)
        return dists.max()
    else:
        return eval(f'np.{method}(dists)')


def calc_zeb_reg(region_lab, img, seg, dist_func, cap_min=True, method='max', r=1):
    """Calculates the Zeb score (contrast) for a particular region.

    For a given region label, this will calculate the Zeb score for the region.
    Returns the score for the region.

    Args:
        region_lab (int): label for a region in seg
        img (ndarray): original image (3D array)
        seg (ndarray): segmentation map (2D array)
        dist_func (str or function): same parameter for scipy pdist function
        cap_min (bool, optional): if true, will make the minimum possible score be 0 (the way the original paper works). Defaults to True.
        method (str, optional): numpy function or 'rob'. Defaults to 'mean'.
        r (int, optional): if method is 'rob', then returns the rth maximum distance. Defaults to 1.

    Returns:
        float: region score
    """
    I_cond = seg == region_lab
    E_cond = seg != region_lab
    boarder = segmentation.find_boundaries(I_cond, mode='inner')
    region = np.argwhere(I_cond)

    I_i = []
    E_i = []
    for s in region:
        I_neighbor = get_cond_neighbor(s[0], s[1], img, I_cond)
        I_i.append(get_contrast(I_neighbor, dist_func, method, r))
        if boarder[s[0], s[1]] == True:
            E_neighbor = get_cond_neighbor(s[0], s[1], img, E_cond)
            E_i.append(get_contrast(E_neighbor, dist_func, method, r))

    if len(E_i) == 0:
        return -inf

    I_i = np.mean(I_i)
    E_i = np.mean(E_i)

    if cap_min:
        if 0 < I_i < E_i:
            C_i = 1 - I_i / E_i
        elif I_i == 0:
            C_i = E_i
        else:
            C_i = 0
    else :
        C_i = 1 - I_i / E_i
    r_score = C_i * np.sum(I_cond) / seg.size
    return r_score


def calc_zeb(img, seg, dist_func, cap_min=True, method='max', r=1, n_proc=8):
    """Calculates the Zeb score.

    For an image (img) and segmentation (seg), this calculates the Zeb score.
    This uses a Pool to calculate each region contrast, then takes the weighted average.

    Args:
        img (ndarray): original image (3D array)
        seg (ndarray): segmentation map (2D array)
        dist_func (str or function): same parameter as scipy pdist function
        n_proc (int, optional): number of processes to spawn. Defaults to 8.
        cap_min (bool, optional): if true, will make the minimum possible score be 0 (the way the original paper works). Defaults to True.
        method (str, optional): numpy function or 'rob'. Defaults to 'mean'.
        r (int, optional): if method is 'rob', then returns the rth maximum distance. Defaults to 1.

    Returns:
        float: total score
    """
    with Pool(n_proc) as p:
        labs = np.unique(seg)
        kwargs = {'img': img,
                  'seg': seg,
                  'dist_func': dist_func,
                  'cap_min': cap_min,
                  'method': method,
                  'r': r}
        res = p.map(partial(calc_zeb_reg, **kwargs), labs)
        p.close()
        p.join()
        res = list(res)
    C_all = np.sum(res)
    return C_all


def calc_zeb_channel(c, img, seg, dist_func, cap_min=True, method='max', r=1, n_proc=8):
    """Calculates the zeb for a single channel

    This wraps calc_zeb to calculate the zeb score for channel c.
    First returns the final score C_all,
    second returns all of the I scores (ordered by np.unique(seg)),
    third returns all of the E scores (same ordering),
    fourth returns all of the C scores (same ordering).

    Args:
        c (int): Which channel to calculate (assumes channel axis is 2).
        img (ndarray): original image (3D array)
        seg (ndarray): segmentation mask (2D array)
        dist_func (str or function): same argument as scipy pdist function.
        cap_min (bool, optional): if true, will make the minimum possible score be 0 (the way the original paper works). Defaults to True.
        method (str, optional): numpy function or 'rob'. Defaults to 'mean'.
        r (int, optional): if method is 'rob', then returns the rth maximum distance. Defaults to 1.
        n_proc (int, optional): Number of processes for calc_zeb. Defaults to 8.

    Returns:
        float: score for the channel
    """
    return calc_zeb(img[:, :, c], seg, dist_func, cap_min, method, r, n_proc)
