from functools import partial

import numpy as np
from multiprocess.pool import Pool
from scipy.linalg import sqrtm
from skimage.segmentation import find_boundaries
from spectral.algorithms import mean_cov


def _get_expanded_region(img, cond, r, min_r, mask):
    """Find the region with additional boundary pixels

    This is similar to how SUQ finds expanded regions.
    Cond should be a bool matrix, where r additional
    layers will be added to cond.
    If r is None, the it will keep adding layers until
    there are atleast img.shape[2] (number of spectral channels)
    samples to help in calculating the covariance matrix.

    Args:
        img (ndarray): 3D image matrix with spectral channels on last axis
        cond (ndarray): 2D boolean matrix with true for the region to grow
        r (int): number of radii/layers to add to cond
        mask (ndarray): boolean array, True where to mask pixels
    Returns:
        ndarray: boolean matrix like cond but with additional layers
    """
    cond[mask] = False
    if r is None:
        i = 0
        cur = np.sum(cond)
        while cur < img.shape[2] or i < min_r:
            edge_e = find_boundaries(cond, mode='outer')
            cond = np.logical_or(cond, edge_e)
            cond[mask] = False
            cur = np.sum(cond)
            i += 1
    else:
        for radii in range(r):
            edge_e = find_boundaries(cond, mode='outer')
            cond = np.logical_or(cond, edge_e)
            cond[mask] = False
    cond[mask] = False
    region = np.argwhere(cond)
    pixels = np.array([img[s[0], s[1], :] for s in region])
    return cond, pixels


def _get_cov_ave(img, cond):
    """Gets the covariance matrix and mean spectra for a region

    Cond should be a boolean matrix, true for the region of interest,
    the pixels to find the covariance and mean of.

    Args:
        img (ndarray): 3D image matrix with spectral channels on last axis.
        cond (ndarray): boolean matrix, true for the pixels to use.

    Returns:
        tuple: covaraince matrix, mean spectra
    """
    mean, cov, nsamples = mean_cov(img, cond, index=True)
    return cov, mean


def _whiten_pixel(pixel, mean, isr_cov):
    """'whitens' a single pixel

    Given a pixel, a mean, and inverse square root covariance matrix,
    return the whitened version of the pixel as:
    isr_cov * (pixel - mean),
    where isr_cov is the inverse square root of the covariance matrix.

    Args:
        pixel (ndarray): array of shape (channels,).
        mean (ndarray): array of shape (channels,).
        isr_cov (ndarray): matrix of shape (channels, channels).

    Returns:
        ndarray: whitened pixel of shape (channels,).
    """
    pixel = pixel - mean
    return np.matmul(isr_cov, pixel)


def _find_invroot_cov(reg, img, seg, radii, min_r, mask):
    """Finds the inverse square root covariance matrix

    For a region, image, and segment, and number of layers,
    this finds the inverse square root of the covariance matrix
    of all the pixels in the region plus radii number of layers.
    Returns first the region that was used,
    second the inverse square root covariance matrix
    third the mean spectra for the region.

    Args:
        reg (int): label for the region of interest.
        img (ndarray): 3D matrix image with spectral channels on the last axis.
        seg (ndarray): 2D segmentation matrix.
        radii (int): number of layers to include in covariance calculation.

    Returns:
        tuple: region, inverse-square-root covariance matrix, mean spectra.
    """
    cond = seg == reg
    cond, pixels = _get_expanded_region(img, cond, radii, min_r, mask)
    cov, mean = _get_cov_ave(img, cond)
    i_cov = np.linalg.inv(cov)
    isr_cov = sqrtm(i_cov)
    return reg, isr_cov, mean


def _find_whiten_pixel(info, img, means, isr_covs):
    """Whitens a given pixel

    info is an entry from np.ndenumerate, so a tuple with
    the coordinate tuple first, and region label second.
    means is a dict with the key being region labels, values
    being mean spectra for that region.
    isr_covs is the same but for inverse square root covariance matrices.
    Return the info, and the whitened pixel.

    Args:
        info (tuple): first is (x, y), second is region label.
        img (ndarray): 3D image matrix with spectral channels on last axis.
        means (dict): keys are region labels, values are mean spectra.
        isr_covs (dict): keys are region labels, values are matries.

    Returns:
        tuple: info, whitened pixel
    """
    idx, r = info
    if r not in means.keys():
        return info, img[idx[0], idx[1], :]
    mean = means[r]
    isr_cov = isr_covs[r]
    return info, _whiten_pixel(img[idx[0], idx[1], :], mean, isr_cov)


def whiten_img(img, seg=None, regs=[], r=None, crop_box={}, n_thread=10, min_r=3, roi_mask=None):
    """whitens an image depending on the segmentation

    Given and image and segmentation of the image, this will locally whiten
    each region using an extra r layers of pixels around each region to
    calculate the covariance and mean.
    Image is a 3D matrix with spectral bands on the third axis, seg is a segmentation
    matrix with integer labels for each region, regs are the regions of interest (if
    [] then will calculate for all regions), and r is the number of additional layers
    to use when calculating mean and cov.
    If seg=None, then it will globally whiten the image (use all pixels to find mean and cov).
    crop_box can be a dict with keys xmin, xmax, ymin, ymax (defaults to take up whole image),
    this way you can reduce calculations by looking at a specific region of the image.
    Note that if you crop the region, only pixels in the cropped image will be used in expanding the regions
    pixels when calculating covariance matrix, and some regions will only have some pixels
    visible in the cropped region but will still have the covariance calculated (these truncated
    regions should not be used).
    Returns the whitened image, as well as the means and inverse square root covariance matrices
    as dictionaries with the regs as keys.
    Note about the threads, it appears finding the inverse is multiprocessed by numpy, so you
    don't need to have too many threads (in fact when finding the covs we use n_thread/2 threads).
    If r is None, then each layer will add at least min_r number of layers, and if the region is
    small, it will continue to add layers until there are at least as many pixels in the region
    as there are spectral channels.

    Args:
        img (ndarray): 3D image matrix with spectral channels on third axis
        seg (ndarray, optional): 2D segmentation matrix for local whitening. Defaults to None.
        regs (list, optional): regions to whiten. Defaults to [].
        r (int, optional): number of overlapping layers to calculate cov and mean. Defaults to None.
        crop_box (dict, optional): see description. Defaults to {}.
        n_thread (int, optional): number of threads. If None, will not use pool (for debugging). Defaults to 10.
        min_r (int, optional): the minimum number of layers to add per region if r=None. Defaults to 3.
        roi_mask (ndarray, optional): a boolean array with True for pixels to mask. Defaults to None.

    Returns:
        tuple: whitened image, means, inverse square root covariances
    """
    if seg is None:
        seg = np.ones((img.shape[0], img.shape[1]))
    if len(regs) == 0:
        regs = np.unique(seg)
    if roi_mask is None:
        roi_mask = np.full_like(seg, False, dtype=bool)

    if len(crop_box.keys()) > 0:
        t_bbox = {}
        t_bbox['xmin'] = 0
        t_bbox['xmax'] = img.shape[0]
        t_bbox['ymin'] = 0
        t_bbox['ymax'] = img.shape[1]
        crop_box = {**t_bbox, **crop_box}
        xmin, xmax = crop_box['xmin'], crop_box['xmax']
        ymin, ymax = crop_box['ymin'], crop_box['ymax']
        img = img[xmin:xmax, ymin:ymax, :]
        seg = seg[xmin:xmax, ymin:ymax]

    means = {}
    isr_covs = {}

    # print('Finding Covariance Matrices...')
    if n_thread is not None:
        with Pool(int(n_thread / 2)) as p:
            args = {'img': img, 'seg': seg, 'radii': r,
                    'min_r': min_r, 'mask': roi_mask}
            res = p.map(partial(_find_invroot_cov, **args), regs)
            p.close()
            p.join()
            res = list(res)

        for l, isr_cov, mean in res:
            means[l] = mean
            isr_covs[l] = isr_cov

        with Pool(n_thread) as p:
            args = {'img': img, 'means': means, 'isr_covs': isr_covs}
            res = p.map(partial(_find_whiten_pixel, **args),
                        np.ndenumerate(seg))
            p.close()
            p.join()
            res = list(res)
    else:
        res = []
        for i in regs:
            res.append(_find_invroot_cov(i, img, seg, r, min_r, roi_mask))

        for l, isr_cov, mean in res:
            means[l] = mean
            isr_covs[l] = isr_cov

        res = []
        for i in np.ndenumerate(seg):
            res.append(_find_whiten_pixel(i, img, means, isr_covs))

    new_img = img.copy()
    for l, pixel in res:
        idx, l = l
        new_img[idx[0], idx[1], :] = pixel
    return -1 * new_img, means, isr_covs


def roi_segment_stats(key, roi_dict, seg):
    """Finds what segments this roi is in

    Given an roi_dict, and key for that roi_dict, and a segmentation,
    this will find what segments the roi resides in.
    As well as the number of roi pixels, number of segment pixels,
    and the proportion of the segments are roi pixels.
    Returns tuple, first the regions,
    second number of roi pixels,
    third the total number of pixels,
    fourth the proportion of roi pixels (second/third)

    Args:
        key (str): key for the roi_dict
        roi_dict (dict): roi_dict
        seg (ndarray): segmentation matrix.

    Returns:
        tuple: see above.
    """
    regs = np.unique(seg[roi_dict[key]['IMG_MASK']])
    n_roi = np.sum(roi_dict[key]["IMG_MASK"])
    total = 0
    for r in regs:
        total += np.sum(seg == r)
    return regs, n_roi, total, n_roi / total


def standardize_spectra(x, n_dim=128):
    """standardizes a spectrum

    This is how the model standardizes the spectra
    before classification.
    n_dim is the number of spectral channels

    Args:
        x (ndarray): one dimensional array of spectral values (shape like (128,)).
        n_dim (int, optional): number of spectral channels. Defaults to 128.

    Returns:
        ndarray: standardized spectra.
    """
    if x.ndim == 1:
        x = x[np.newaxis, np.newaxis, :, np.newaxis]
    if x.ndim == 2:
        x = x[:, np.newaxis, :, np.newaxis]
    x = x/np.std(x, axis=2, keepdims=True)
    x = x-np.mean(x, axis=2, keepdims=True)
    return x.reshape(n_dim,)
