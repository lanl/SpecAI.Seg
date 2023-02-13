import warnings
from functools import partial
from multiprocessing.pool import Pool

import numpy as np
from numpy.random import default_rng
from skimage.segmentation import find_boundaries

from ..metrics.distance import cos
from ._uq_metric import UQMetric


class suqsES(UQMetric):
    """Default implementation of Segmentation Uncertainty Quantification

    This implementation uses the inverse SD for the Q score.
    Q is found by (1/n sum d(x, x_bar)^2)^-1, where d is the dist_func
    The boundary perturbation method is Random Radial Inclusion/Exclusion,
    which randomly adds or removes boundary pixels within a radius r, with probability p.
    The quantification equation is Ratio of Extended/Shrunken Q scores,
    that is SUQ = 1/2(Q(e)/Q(o) + Q(o)/Q(s)), where o, e, and s are the origignal,
    extended, and shrunken regions.
    """

    def __init__(self, img, seg, dist_func=cos, n=1, p=1, r=1, min_pixels=2, size_penalty=False, seed=42069, n_thread=16,
                 disable_min_check=False):
        """Set parameters for SUQ score calculation.

        See docstring for suqsES for more details. Some notes:
        Size penalty is recommended to be False.
        This code uses multiprocessing when calulating SUQ for all regions.

        Args:
            img (ndarray or dict): original image, or dict with key 'img' with value of 3D array.
            seg (ndarray): segmentation mask (2D array)
            dist_func (func, optional): Distance metric used to calculate SD. Defaults to cos.
            n (int, optional): Number of simulations to conduct per region. Defaults to 1.
            p (int, optional): Probability of inclustion or exclusion in extended/shrunken regions. Defaults to 1.
            r (int, optional): Number of boundary radii to look to include/exclude. Defaults to 1.
            min_pixels (int, optional): Each shrunken boundary will have at least this many pixels. Defaults to 2.
            size_penalty (bool, optional): Apply size penalty for large increases/decreases in pixel count. Defaults to False.
            seed (int, optional): Seed of random number generator. Defaults to 42069.
            n_thread (int, optional): Number of processes to spin up. Defaults to 16.
            disable_min_check (bool, optional): If True, then it wont check to make sure there are at least min_pixels in all regions. Defaults to False.
        """
        super().__init__(img, seg)

        self.dist_func = dist_func
        if n < 1:
            warnings.warn('n is less than 1; making n=1')
            n = 1
        self.n = int(n)
        if p < 0:
            warnings.warn('p is less than 0; making p=0')
            p = 0
        elif p > 1:
            warnings.warn('p is greater than 1; making p=1')
            p = 1
        self.p = p
        if r < 1:
            warnings.warn('r is less than 1; making r=1')
            r = 1
        self.r = int(r)
        self.min_pixels = int(min_pixels)
        self.disable_min_check = disable_min_check
        self.set_info(seg)

        self.size_penalty = size_penalty
        self.seed = seed
        self.n_thread = n_thread

    def evaluate(self, img=None, seg=None):
        """Calculates SUQ for all regions in seg

        The first returned value is a list of all scores, ordered by np.unique(seg).
        The second return value is a dict with region score pairs

        Args:
            img (ndarray or dict): original image, or dict with key 'img' with value of 3D array.
            seg (ndarray): segmentation mask (2D array)

        Returns:
            tuple: list of all scores, dict with region score pairs
        """
        img, seg = super().evaluate(img, seg)
        labs = np.unique(seg)
        res = []
        with Pool(self.n_thread) as p:
            args = {'img': img, 'seg': seg}
            res = p.map(partial(self.get_uq_reg, **args), labs)
            p.close()
            p.join()
            res = list(res)
        uqs = {}
        uql = []
        for i in range(len(labs)):
            uqs[labs[i]] = res[i]
            uql.append(res[i][0])
        return uql, uqs

    def _default_QE(self, img, seg_0, seg_e, seg_s, dist_func, size_penalty, lab=None):
        """Calculate SUQ using a quantification equation

        This uses ratio of extended/shrunken regions to find score.
        If finds 1/2(Q(e)/Q(o) + Q(o)/Q(s)),
        With size penalty: 1/2(|e|Q(e)/|o|Q(o) |o|Q(o)/|s|Q(s))

        Args:
            img (ndarray): The image
            seg_0 (ndarray): Array with True for pixels in the original segmentation.
            seg_e (ndarray): Array with True for pixels in the enlarged segmentation.
            seg_s (ndarray): Array with True for pixels in the shrunken segmentation.
            dist_func (func): distance function to be passed to the Q score for finding SD
            size_penalty (bool): If true then apply the additional size penalties.
            lab (bool or int, optional): If None, then uses True as the region label for _default_QS. Defaults to None.

        Returns:
            float: SUQ score
        """
        Q_0 = self._default_QS(img, seg_0, dist_func, lab)
        Q_e = self._default_QS(img, seg_e, dist_func, lab)
        Q_s = self._default_QS(img, seg_s, dist_func, lab)
        if size_penalty:
            R_0 = np.sum(seg_0)
            R_e = np.sum(seg_e)
            R_s = np.sum(seg_s)
            a = (R_e / R_0) * (Q_e / Q_0)
            b = (R_0 / R_s) * (Q_0 / Q_s)
            return (a + b) / 2
        else:
            return ((Q_e / Q_0) + (Q_0 / Q_s)) / 2

    def _default_QS(self, img, seg, dist_func, lab=None):
        """Calculates Quality Score

        Specifically finds inverse SD as the mean of the squared
        distance between each spectra and the average spectra
        found from all the pixels.
        Note that you need at least 2 unique pixels to find this.

        Args:
            img (ndarray): The image
            seg (ndarray): Array showing which pixels are in which groups
            dist_func (func): function to calculate distance between spectra and average spectra
            lab (int or None, optional): gets pixels according to this label. If none then gets True pixels. Defaults to None.

        Returns:
            float: Inverse SD
        """
        if lab is None:
            pixels = self._get_pixels(True, img, seg)
        else:
            pixels = self._get_pixels(lab, img, seg)
        ave = np.mean(pixels, axis=0)
        dists = np.array([dist_func(x, ave) for x in pixels])
        sd = np.sqrt(np.mean(np.square(dists)))
        if sd == 0:
            raise RuntimeError(
                f'0 SD in region {lab}. Need at least 2 unique pixels!')
        return 1 / sd

    def _get_pixels(self, lab, img, seg):
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

    def _get_new_boundary(self, cond, rng, indices_e, indices_s, p, min_pixels, indices_og):
        """Finds enlarged and shrunken boundaries

        Makes boolean matrix for the original segment, and for new larger and smaller segments.
        It finds the larger regions by randomly adding pixels within r layers of the original seg.
        Cond should be matrix with True where the original segment is.
        indices_e and indices_s should be the (x,y) of points to randomly sample from (see _get_indices).
        If the randomly removed pixels for indices_s is less than min_pixels, then
        min_pixels will be randomly selected from the indices_og.
        Returns a tuple with first the enlarged region, then the shrunken region (boolean matrices)

        Args:
            cond (ndarray): Boolean matrix where the original segment is
            rng (numpy.random._generator.Generator): random number generator (need a random and choice function)
            indices_e (ndarray): Result of np.argwhere for the pixels considered to add to region
            indices_s (ndarray): Result of np.argwhere for the pixels considered to remove from the region
            p (float): probability of adding pixel to E, and removing pixel to S
            min_pixels (int): the minimum number of pixels to have in the shrunken region
            indices_og (ndarray): Result of np.argwhere for the pixels in the original segment

        Returns:
            ndarray, ndarray: tuple of matrices containing the new enlarged and shrunken regions.
        """
        cond_e = cond.copy()
        cond_s = cond.copy()
        rng_e = rng.random(indices_e.shape[0])
        rng_s = rng.random(indices_s.shape[0])
        for i in range(len(rng_e)):
            x = indices_e[i, 0]
            y = indices_e[i, 1]
            cond_e[x, y] = rng_e[i] < p
        for i in range(len(rng_s)):
            x = indices_s[i, 0]
            y = indices_s[i, 1]
            cond_s[x, y] = rng_s[i] > p
        if np.sum(cond_s) < min_pixels:
            idxs = rng.choice(
                indices_og.shape[0], size=min_pixels, replace=False)
            for i in idxs:
                x = indices_og[i, 0]
                y = indices_og[i, 1]
                cond_s[x, y] = True
        return cond_e, cond_s

    def _get_indices(self, new_info, new_lab, r):
        """Finds the enlarged and shrunken boundary pixels

        This will find the potental pixels for inclusion or exclusion of the region.
        Returns 1. boolean matrix, True where the original segment is
        2. indices of pixels within r radii outside original segment boundary
        3. indices of pixels within r radii inside original segment boundary
        4. indices of pixels in the original segment

        Args:
            new_info (ndarray): segmentation array
            new_lab (int): label/number of the region of consideration
            r (int): number of radii to grow/shrink

        Returns:
            4-tuple: see above
        """
        cond = new_info == new_lab
        boundary_e = cond.copy()
        boundary_s = cond.copy()
        for radii in range(r):
            edge_e = find_boundaries(boundary_e, mode='outer')
            edge_s = find_boundaries(boundary_s, mode='inner')
            boundary_e = np.logical_or(boundary_e, edge_e)
            boundary_s = np.logical_and(boundary_s, np.logical_not(edge_s))
        boundary_e = np.logical_and(boundary_e, np.logical_not(cond))
        boundary_s = np.logical_and(cond, np.logical_not(boundary_s))
        indices_e = np.argwhere(boundary_e)
        indices_s = np.argwhere(boundary_s)
        indices_og = np.argwhere(cond)
        return cond, indices_e, indices_s, indices_og

    def _uq_reg(self, lab, img, seg, n, p, r, dist_func, size_penalty, min_pixels, seed=420):
        """Calculate SUQ for a region

        See class docstring for details on what this number is.
        Returns 1. mean SUQ score
        2. list of all SUQ scores for each simulation.

        Args:
            lab (tuple ints): tuple of all region labels to be considered in calculating score.
            img (ndarray or dict): original image, or dict with key 'img' with value of 3D array.
            seg (ndarray): segmentation mask (2D array)
            n (int): number of simulations of E/S regions to do
            p (float): probability of including/excluding pixels for enlarged/shrunken regions
            r (int): number of radii to look to expand/shrink region
            dist_func (function): distance function to use in calculating SD
            size_penalty (bool): Wether or not to apply size penalty to final SUQ score
            min_pixels (int): minimum number of pixels to have in a shrunken region (should be 2 for SD).
            seed (int, optional): seed for RNG. Defaults to 420.

        Returns:
            tuple: average score, list of scores
        """
        img, seg = super().evaluate(img, seg)
        new_info = seg.copy()
        new_lab = lab[0]
        for l in lab:
            new_info = np.where(seg == l, new_lab, new_info)

        # Finds the enlarged and shrunked boundaries
        cond, indices_e, indices_s, indices_og = self._get_indices(
            new_info, new_lab, r)

        # Randomly includes/excludes boundary pixels
        rng = default_rng(seed)
        res = []
        for rep in range(n):
            cond_e, cond_s = self._get_new_boundary(
                cond, rng, indices_e, indices_s, p, min_pixels, indices_og)
            res.append(self._default_QE(img, cond, cond_e,
                       cond_s, dist_func, size_penalty))
        return np.mean(res), res

    def get_uq_reg(self, *reg, img=None, seg=None):
        """Gets the UQ for the regions

        If you specify multiple numbers for reg, it will find
        SUQ as if they are combined.
        Uses all the parameters original set when initialized.
        You can specify a new img and seg if you want.
        Return the average SUQ, and list of all SUQ values from simulations

        Args:
            *reg (ints): labels of the regions of interest.
            img (ndarray or dict): original image, or dict with key 'img' with value of 3D array.
            seg (ndarray): segmentation mask (2D array)

        Returns:
            tuple: average SUQ, all SUQ scores
        """
        v = self._uq_reg(reg, img, seg, self.n, self.p, self.r,
                         self.dist_func, self.size_penalty, self.min_pixels, self.seed)
        return v

    def _get_quality_score(self, *reg, img=None, seg=None):
        """Finds just the quality score for region

        If multiple regions are specified, it returns the quality 
        score if they were all the same region.
        Note this does not do any simulation, just the quality
        score for the original segment (or merged segments).

        Args:
            img (ndarray or dict): original image, or dict with key 'img' with value of 3D array.
            seg (ndarray): segmentation mask (2D array)

        Returns:
            float: inverse SD of the regions of interest
        """
        img, seg = super().evaluate(img, seg)
        new_info = seg.copy()
        new_lab = reg[0]
        for l in reg:
            new_info = np.where(seg == l, new_lab, new_info)
        return self._default_QS(img, new_info, self.dist_func, new_lab)

    def get_q_score_reg(self, *reg, img=None, seg=None):
        """Finds just the quality score for region

        If multiple regions are specified, it returns the quality 
        score if they were all the same region.
        Note this does not do any simulation, just the quality
        score for the original segment (or merged segments).

        Args:
            img (ndarray or dict): original image, or dict with key 'img' with value of 3D array.
            seg (ndarray): segmentation mask (2D array)

        Returns:
            float: inverse SD of the regions of interest
        """
        return self._get_quality_score(*reg, img=img, seg=seg)

    def get_e_s_boundaries(self, *reg, img=None, seg=None):
        """Returns a single simulated extended/shrunken regions

        Will use the same parameters as where initialized, but
        only does 1 simulation.
        Can specify multiple regions to be considered as merged.
        Returns 1. matrix with True where the extended region is
        2. matrix with True where the shrunken region is
        3. quality score for enlarged region
        4. quality score for shrunken region
        5. quality score for the original region
        6. final SUQ score

        Args:
            img (ndarray or dict): original image, or dict with key 'img' with value of 3D array.
            seg (ndarray): segmentation mask (2D array)

        Returns:
            6-tuple: see above
        """
        img, seg = super().evaluate(img, seg)
        new_info = seg.copy()
        new_lab = reg[0]
        for l in reg:
            new_info = np.where(seg == l, new_lab, new_info)
        rng = default_rng(self.seed)
        cond, indices_e, indices_s, indices_og = self._get_indices(
            new_info, new_lab, self.r)
        cond_e, cond_s = self._get_new_boundary(
            cond, rng, indices_e, indices_s, self.p, self.min_pixels, indices_og)
        score = self._default_QE(
            img, cond, cond_e, cond_s, self.dist_func, self.size_penalty)
        q_e, q_s = self._default_QS(img, cond_e, self.dist_func), self._default_QS(
            img, cond_s, self.dist_func)
        q_o = self._default_QS(img, cond, self.dist_func)
        return cond_e, cond_s, cond, q_e, q_s, q_o, score

    def set_info(self, info):
        """Function to update the segmentation

        Args:
            info (ndarray): new segmentation array

        Raises:
            RuntimeError: Makes sure each region has at least the minimum number of pixels.
        """
        if not self.disable_min_check:
            if info is not None:
                bin_count = np.bincount(info.flatten())
                bin_count = np.where(bin_count == 0, np.nan, bin_count)
                if np.nanmin(bin_count) < self.min_pixels:
                    raise RuntimeError(
                        'Smallest region is smaller than minimum pixels')
                else:
                    self.info = info
                    self._change = True
        else:
            self.info = info
            self._change = True

    def set_r(self, r):
        """update the r value

        Args:
            r (int): new number of radii for calculation
        """
        self.r = r
        self._change = True

    def set_n(self, n):
        """update the n value

        Args:
            n (int): new number of simulations for calculation
        """
        self.n = n
        self._change = True

    def set_p(self, p):
        """update the p value

        Args:
            p (float): new number of probability for calculation
        """
        self.p = p
        self._change = True
