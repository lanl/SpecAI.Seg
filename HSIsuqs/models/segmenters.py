import warnings
from abc import ABC, abstractmethod
from functools import partial
from multiprocessing.pool import Pool

import graph_tool as gt
import numpy as np
from graph_tool.topology import label_components
from skimage import morphology, segmentation
from skimage._shared.filters import gaussian
from skimage._shared.utils import _supported_float_type
from skimage.segmentation._felzenszwalb_cy import _felzenszwalb_cython
from skimage.segmentation._quickshift_cy import _quickshift_cython
from skimage.util import img_as_float
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import pairwise_distances
from sklearn.neighbors import kneighbors_graph
from tqdm import tqdm

from ..metrics import distance as dist
from ..uq.suqs import suqsES
from ._slic_wrap import slic


def label_clusters(mat):
    """Sequentially labels distinct clusters in segmentation

    The input can be either a True/False (foreground/background) segmentation,
    or a matrix where every group/cluster shares the same value.
    This will essentially relabel all the clusters starting from 0.
    If a 3D array is passed in it will label each channel separately.

    Args:
        mat (ndarray): 2D/3D array with raw segmentation (or could be labeled)

    Returns:
        ndarray: 2D/3D array with labels for each cluster
    """
    if len(mat.shape) == 3:
        results = np.zeros_like(mat)
        for c in tqdm(range(mat.shape[2])):
            results[:, :, c] = label_clusters(mat[:, :, c])
        return results
    else:
        return morphology.label(mat, connectivity=1, background=-1)


def _label_graph(mat):
    """Labels segments, similar to label_clusters.

    Like label_clusters, this labels each cluster in a segmentation starting from 0.
    This uses a graph, and labels based on connected components.
    It also returns the counts for each component for use to remove small groups.

    Args:
        mat (ndarray): a 2d matrix to label clusters

    Returns:
        tuple: the segmentation map, and the counts for each segment
    """
    g = gt.Graph(directed=False)
    verts = np.empty_like(mat, dtype=gt.Vertex)
    for r in range(mat.shape[0]):
        for c in range(mat.shape[1]):
            v = g.add_vertex()
            verts[r, c] = v

    for r in range(mat.shape[0]):
        for c in range(mat.shape[1]):
            lab = mat[r, c]
            lu = r - 1 >= 0
            ld = r + 1 < verts.shape[0]
            ll = c - 1 >= 0
            lr = c + 1 < verts.shape[1]

            if lu and mat[r - 1, c] == lab:
                g.add_edge(verts[r - 1, c], verts[r, c])
            if ld and mat[r + 1, c] == lab:
                g.add_edge(verts[r + 1, c], verts[r, c])
            if ll and mat[r, c - 1] == lab:
                g.add_edge(verts[r, c - 1], verts[r, c])
            if lr and mat[r, c + 1] == lab:
                g.add_edge(verts[r, c + 1], verts[r, c])

    comp, counts = label_components(g, directed=False)
    i = 0
    res = np.empty_like(mat)
    for r in range(mat.shape[0]):
        for c in range(mat.shape[1]):
            res[r, c] = comp[i]
            i += 1
    return res.astype(int), counts


def _remove_singletons(seg, min_size, relabel=True):
    """Given a segmentaiton, it removes merges small clusters.

    Any segment smaller than min_size will get merged with the neighbor with the most touching edges.

    Args:
        seg (ndarray): 2d array of labeled segments.

    Returns:
        ndarray: relabeled segmentation with small clusters merged with nearby clusters.
    """
    if relabel:
        seg, counts = _label_graph(seg)
        for i in range(len(counts)):
            if counts[i] < min_size:
                temp = seg == i
                neigh = seg[segmentation.find_boundaries(temp, 2, 'outer')]
                new_lab = np.bincount(neigh).argmax()
                seg[seg == i] = new_lab
        return label_clusters(seg)
    else:
        for i in np.unique(seg):
            temp = seg == i
            if np.sum(temp) < min_size:
                neigh = seg[segmentation.find_boundaries(temp, 2, 'outer')]
                new_lab = np.bincount(neigh).argmax()
                seg[seg == i] = new_lab
        new_lab = 1
        new_seg = seg.copy()
        for i in np.unique(seg):
            new_seg[seg == i] = new_lab
            new_lab += 1
        return new_seg



def _find_smallest(seg):
    """Finds the size of smallest region

    Args:
        seg (ndarray): segmentation array

    Returns:
        int: size of smallest region
    """
    counts = np.bincount(seg.flatten())
    counts = np.where(counts == 0, np.nan, counts)
    return np.nanmin(counts)


def merge_smallest(seg, min_size):
    """Merge all regions untill the smallest is min_size

    Args:
        seg (ndarray): segmentation matrix
        min_size (int): the desired minimum size of regions

    Returns:
        ndarray: new relabeled segmentation matrix
    """
    curr_min = _find_smallest(seg)
    new_seg = seg.copy()
    while curr_min < min_size:
        new_seg = _remove_singletons(new_seg, min_size)
        curr_min = _find_smallest(new_seg)
    return label_clusters(new_seg)


class Segmenter(ABC):
    """Abstract class for all segmentation algorithms

    Every segmentation algorithm should extend this class.
    """

    @abstractmethod
    def segment(self, img=None):
        """This method will segment the image.

        Args:
            img (ndarray, optional): A 3D image to segment. Defaults to None.

        Returns:
            ndarray: 2D array with segmented regions labeled.
        """
        if img is None:
            return self.img
        return img

    def get_segmentation(self, labels=True):
        """Returns segmentation, finds segmentation if needed.

        Args:
            labels (bool, optional): Labels the segments starting from 0, or returns raw output. Defaults to True.

        Returns:
            ndarray: 2D array with segments labeled
        """
        if self.seg is None or self._change:
            self.seg = self.segment()
        if labels:
            return label_clusters(self.seg)
        return self.seg

    def __init__(self, img):
        """Sets up the common variables.

        Args:
            data (dict or ndarray): a dictionary with at least an 'img' key with 3D hyperspectral image, or just the 3D array.
        """
        self.set_img(img)

        self.seg = None
        self._change = True

    def set_img(self, img):
        """Sets the image

        Args:
            img (dict or ndarray): a dictionary with at least an 'img' key with 3D hyperspectral image, or just the 3D array.

        Returns:
            ndarray: the img
        """
        if isinstance(img, dict):
            self.img = img['img']
        elif isinstance(img, np.ndarray):
            self.img = img
        else:
            self.img = None
        self._change = True
        return self.img


class Felzenszwalb(Segmenter):
    """Multi-spectral implementation of the Felzenszwalb algorithm.

    This is a wrapper/alteration of the code from skimage.segment.Felzenszwalb.
    All that is changed is it suppresses the warning about using more than 3 channels.

        Typical usage example:

        indianPines = datasets.get_data('IndianPines')
        seg = Felzenszwalb(indianPines, scale=100).get_segmentation()
    """

    def __str__(self) -> str:
        return 'Felzenszwalb'

    def __init__(self, img, scale=1000.0, sigma=0.8, min_size=20):
        """Sets additional parameters for Felzenszwalb.

        The available parameters to set are 'scale', 'sigma', 'min_size'.
        See the skimage documentation for what these parameters do.

        Args:
            img (dict or ndarray): a dictionary with at least an 'img' key with 3D hyperspectral image, or just the 3D array.
            scale (float, optional): Free parameters. Higher means larger clusters. Defaults to 1000.
            sigma (float, optional): Width (standard deviation) of Gaussian kernel used in preprocessing. Defaults to 0.8.
            min_size (int, optional): The minimum number of pixels in a region. Defaults to 20.
        """
        super().__init__(img)
        self.set_scale(scale)
        self.set_sigma(sigma)
        self.set_min_size(min_size)

    def set_scale(self, scale):
        """Set the scale

        Args:
            scale (float): a new scale parameter
        """
        self.scale = scale
        self._change = True

    def set_sigma(self, sigma):
        """Set a new sigma parameter

        Args:
            sigma (float): new float parameter
        """
        self.sigma = sigma
        self._change = True

    def set_min_size(self, min_size):
        """Set new min_size

        Args:
            min_size (int): new min_size parameter
        """
        self.min_size = min_size
        self._change = True

    def segment(self, img=None):
        """Segments the image using Felzenszwalb algorithm.

        Args:
            img (ndarray, optional): a 3D image to segment. Defaults to None.

        Returns:
            ndarray: segmentation mask
        """
        img = super().segment(img)

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            seg = _felzenszwalb_cython(img, scale=self.scale, sigma=self.sigma,
                                       min_size=self.min_size)
        return _remove_singletons(seg, self.min_size)


class Quickshift(Segmenter):
    """Alteration of skimage's Quickshift segmentor.

    The code wraps skimage's Quickshift function so that it wont complain about more than 3 channels.
    See skimage.segmentation documentation for info about the algorithm and parameters.

        Typical usage example:

        indianPines = datasets.get_data('IndianPines')
        seg = Quickshift(indianPines).get_segmentation()
    """

    def __str__(self) -> str:
        return 'Quickshift'

    def __init__(self, data, ratio=1.0, kernel_size=5, max_dist=10,
                 sigma=0, random_seed=42, channel_axis=-1, min_size=20):
        """Additional parameters for Quickshift algorithm.

        See skimage.segmentation.quickshift documentation for more info on parameters.
        They are 'ratio', 'kernal_size', 'max_dist', 'sigma', 'random_seed', 'channel_axis'.

        Args:
            data (dict or ndarray): a dictionary with at least an 'img' key with 3D hyperspectral image, or just the 3D array.
            ratio (float, optional): Balances color-space proximity and image-space proximity. Higher values give more weight to color-space. Defaults to 1.0.
            kernel_size (float, optional): Width of Gaussian kernel used in smoothing the sample density. Higher means fewer clusters. Defaults to 5.
            max_dist (float, optional): Cut-off point for data distances. Higher means fewer clusters. Defaults to 10.
            sigma (float, optional): Width for Gaussian smoothing as preprocessing. Zero means no smoothing. Defaults to 0.
            random_seed (int, optional): Ranodm seed used for breaking ties. Defaults to 42.
            channel_axis (int, optional): The axis of img corresponding to color channels. Defaults to -1.
            min_size (int, optional): The minimum number of pixels in a region. Defaults to 20.
        """
        super().__init__(data)
        self.set_ratio(ratio)
        self.set_kernel_size(kernel_size)
        self.set_max_dist(max_dist)
        self.return_tree = False
        self.set_sigma(sigma)
        self.set_random_seed(random_seed)
        self.set_channel_axis(channel_axis)
        self.set_min_size(min_size)

    def set_ratio(self, ratio):
        self.ratio = ratio
        self._change = True

    def set_kernel_size(self, kernel_size):
        self.kernel_size = kernel_size
        self._change = True

    def set_max_dist(self, max_dist):
        self.max_dist = max_dist
        self._change = True

    def set_sigma(self, sigma):
        self.sigma = sigma
        self._change = True

    def set_random_seed(self, random_seed):
        self.random_seed = random_seed
        self._change = True

    def set_channel_axis(self, channel_axis):
        self.channel_axis = channel_axis
        self._change = True

    def set_min_size(self, min_size):
        self.min_size = min_size
        self._change = True

    def segment(self, img=None):
        """Segments the image using the Quickshift algorithm.

        Args:
            img (ndarray, optional): 3D image array to segment. Defaults to None.

        Raises:
            ValueError: Makes sure the kernel size is greater of equal to 1.

        Returns:
            ndarray: 2D segmentation map sequentially labels from 0.
        """
        img = super().segment(img)

        img = img_as_float(np.atleast_3d(img))
        float_dtype = _supported_float_type(img.dtype)
        img = img.astype(float_dtype, copy=False)
        img = np.moveaxis(img, source=self.channel_axis, destination=-1)

        if self.kernel_size < 1:
            raise ValueError("`kernel_size' should be >= 1.")

        img = gaussian(img, [self.sigma, self.sigma, 0],
                       mode='reflect', channel_axis=-1)
        img = np.ascontiguousarray(img * self.ratio)

        seg = _quickshift_cython(img, kernel_size=self.kernel_size, max_dist=self.max_dist,
                                 return_tree=self.return_tree, random_seed=self.random_seed)
        return _remove_singletons(seg, self.min_size)


class Slic(Segmenter):
    """Alteration of skimage's SLIC segmentation algorithm.

    The alteration just makes it work with more than 3 channels.
    See skimage.segmentation.slic documentation for more information.

        Typical usage example:

        ksc = datasets.get_data('KSC')
        seg = Slic(ksc, n_segments=250).get_segmentation()
    """

    def __str__(self) -> str:
        return 'Slic'

    def __init__(self, data, n_segments=100, compactness=None, max_num_iter=10,
                 sigma=0, spacing=None, enforce_connectivity=True,
                 min_size_factor=0.5, max_size_factor=3, slic_zero=False,
                 start_label=1, mask=None, channel_axis=-1, min_size=20,
                 dist_func='cos'):
        """Sets additional parameters for the algorithm.

        See skimage slic documentation for more information.
        The params are 'n_segments', 'compactness', 'max_num_iter', 'sigma', 'spacing',
        'enforce_connectivity', 'min_size_factor', 'max_size_factor', 'slic_zero',
        'start_label', 'mask', 'channel_axis'.

        Args:
            data (dict or ndarray): a dictionary with at least an 'img' key with 3D hyperspectral image, or just the 3D array.
            n_segments (int, optional): The (approximate) number of labels in the segmented output image. Defaults to 100.
            compactness (float, optional): Balances color proximity and space proximity.
                Higher values give more weight to space proximity, making superpixel shapes more square/cubic.
                In SLICO mode, this is the initial compactness. This parameter depends strongly on image contrast and on the shapes of objects in the image. 
                We recommend exploring possible values on a log scale, e.g., 0.01, 0.1, 1, 10, 100, before refining around a chosen value.
                Defaults to 0.005 for cos and 1 for euclid.
            max_num_iter (int, optional): Maximum number of iterations of k-means. Defaults to 10.
            sigma (float, optional): Width of Gaussian smoothing kernel for pre-processing for each
                dimension of the image. The same sigma is applied to each dimension in
                case of a scalar value. Zero means no smoothing.
                Note that `sigma` is automatically scaled if it is scalar and
                if a manual voxel spacing is provided (see Notes section). If
                sigma is array-like, its size must match ``image``'s number
                of spatial dimensions. Defaults to 0.
            spacing (array-like of floats, optional): The voxel spacing along each spatial dimension. By default,
                `slic` assumes uniform spacing (same voxel resolution along
                each spatial dimension).
                This parameter controls the weights of the distances along the
                spatial dimensions during k-means clustering. Defaults to None.
            enforce_connectivity (bool, optional): Whether the generated segments are connected or not. Defaults to True.
            min_size_factor (float, optional): Proportion of the minimum segment size to be removed with respect
                to the supposed segment size ```depth*width*height/n_segments```. Defaults to 0.5.
            max_size_factor (float, optional): Proportion of the maximum connected segment size. A value of 3 works
                in most of the cases. Defaults to 3.
            slic_zero (bool, optional): Run SLIC-zero, the zero-parameter mode of SLIC. Defaults to False.
            start_label (int, optional): The labels' index start. Should be 0 or 1. Defaults to 1.
            mask (ndarray, optional): If provided, superpixels are computed only where mask is True,
                and seed points are homogeneously distributed over the mask
                using a k-means clustering strategy. Mask number of dimensions
                must be equal to image number of spatial dimensions. Defaults to None.
            channel_axis (int, optional): If None, the image is assumed to be a grayscale (single channel) image.
                Otherwise, this parameter indicates which axis of the array corresponds
                to channels. Defaults to -1.
            min_size (int, optional): The minimum number of pixels in a region. Defaults to 20.
            dist_func (str, optional): Either euclid or cos (euclid is original skimage implementation, cos is a modified version
                to use spectral cosine distance instead of euclidean distance for the Slic segmentation). Defaults to 'cos'
        """
        super().__init__(data)
        self.set_n_segments(n_segments)
        self.set_dist_func(dist_func)
        self.set_compactness(compactness)
        self.set_max_num_iter(max_num_iter)
        self.set_sigma(sigma)
        self.set_spacing(spacing)
        self.set_enforce_connectivity(enforce_connectivity)
        self.set_min_size_factor(min_size_factor)
        self.set_max_size_factor(max_size_factor)
        self.set_slic_zero(slic_zero)
        self.set_start_label(start_label)
        self.set_mask(mask)
        self.set_channel_axis(channel_axis)
        self.set_min_size(min_size)

    def set_sigma(self, sigma):
        self.sigma = sigma
        self._change = True

    def set_n_segments(self, n_segments):
        self.n_segments = n_segments
        self._change = True

    def set_compactness(self, compactness):
        if self.dist_func == 'cos':
            self.compactness = 0.005
        elif self.dist_func == 'euclid':
            self.compactness = 1.0
        if compactness is not None:
            self.compactness = compactness
        self._change = True

    def set_max_num_iter(self, max_num_iter):
        self.max_num_iter = max_num_iter
        self._change = True

    def set_spacing(self, spacing):
        self.spacing = spacing
        self._change = True

    def set_enforce_connectivity(self, enforce_connectivity):
        self.enforce_connectivity = enforce_connectivity
        self._change = True

    def set_min_size_factor(self, min_size_factor):
        self.min_size_factor = min_size_factor
        self._change = True

    def set_max_size_factor(self, max_size_factor):
        self.max_size_factor = max_size_factor
        self._change = True

    def set_slic_zero(self, slic_zero):
        self.slic_zero = slic_zero
        self._change = True

    def set_start_label(self, start_label):
        self.start_label = start_label
        self._change = True

    def set_mask(self, mask):
        self.mask = mask
        self._change = True

    def set_channel_axis(self, channel_axis):
        self.channel_axis = channel_axis
        self._change = True

    def set_min_size(self, min_size):
        self.min_size = min_size
        self._change = True

    def set_dist_func(self, dist_func):
        self.dist_func = dist_func.lower()
        self._change = True

    def segment(self, img=None):
        """Creates the slic segmentation

        Args:
            img (ndarray, optional): 3D image. Defaults to None.

        Returns:
            ndarray: 2D array segmentation mask.
        """
        img = super().segment(img)

        if self.dist_func == 'cos':
            seg = slic(img, n_segments=self.n_segments, compactness=self.compactness,
                       max_num_iter=self.max_num_iter, sigma=self.sigma, spacing=self.spacing,
                       enforce_connectivity=self.enforce_connectivity, min_size_factor=self.min_size_factor,
                       max_size_factor=self.max_size_factor, slic_zero=self.slic_zero)
        elif self.dist_func == 'euclid':
            seg = segmentation.slic(img, n_segments=self.n_segments, compactness=self.compactness,
                                    max_num_iter=self.max_num_iter, sigma=self.sigma, spacing=self.spacing,
                                    enforce_connectivity=self.enforce_connectivity, min_size_factor=self.min_size_factor,
                                    max_size_factor=self.max_size_factor, slic_zero=self.slic_zero)
        return _remove_singletons(seg, self.min_size)


class Watershed(Segmenter):
    """Watershed segmentation using Robust Color Morphological Gradient.

    The watershed algorithm used is from skimage.segmentation.
    To work with HSI, the Robust Color Morphological Gradient is calculated and passed to the watershed alg.
    Details can be found at https://www.sciencedirect.com/science/article/pii/S003132031000049X
    "Segmentation and classification of hyperspectral images using watershed transformation".
    This implementation has an extra step to merge small segments with a neighboring segment.

        Typical usage example:

        import HSIsuqs.metrics as m
        indianPines = datasets.get_data('IndianPines')
        seg = Watershed(indianPines, markers=250, dist_func=m.dist_kl).get_segmentation()
    """

    def __str__(self) -> str:
        return 'Watershed'

    def __init__(self, data, markers=None, connectivity=1, offset=None, mask=None,
                 compactness=0, watershed_line=False, dist_func='euclidean',
                 window=1, min_size=20, n_proc=8):
        """Sets additional parameters for the watershed algorithm.

        For most parameters see the skimage.segmentation.watershed documentation.
        Other parameters are used to control the RCMG calculation.
        The watershed args are 'markers', 'connectivity', 'offset', 'mask', 'compactness', 'watershed_line',
        the gradient args are 'dist_func', 'window', 'min_size'.

        Args:
            data (dict or ndarray): a dictionary with at least an 'img' key with 3D hyperspectral image, or just the 3D array.
            markers (int, optional): The desired number of markers, or an array marking the basins with the
                values to be assigned in the label matrix. Zero means not a marker. If
                ``None`` (no markers given), the local minima of the image are used as
                markers. Defaults to None.
            connectivity (ndarray, optional): An array with the same number of dimensions as `image` whose
                non-zero elements indicate neighbors for connection.
                Following the scipy convention, default is a one-connected array of
                the dimension of the image. Defaults to 1
            offset (array_like, optional): offset of the connectivity (one offset per dimension). Defaults to None.
            mask (ndarray, optional): Array of same shape as `image`. Only points at which mask == True
                will be labeled. Defaults to None.
            compactness (float, optional): Use compact watershed with given compactness parameter.
                Higher values result in more regularly-shaped watershed basins. (recomend leaving as 0). Defaults to 0.
            watershed_line (bool, optional): If watershed_line is True, a one-pixel wide line separates the regions
                obtained by the watershed algorithm. The line has the label 0. Defaults to False.
            dist_func (function, optional): a distance function to calculate the similarity (distance) between two spectra.
                It is passed to sklearn pairwise distance, so you can also pass any of those options. Defaults to 'euclidean'.
            window (int, optional): the size of window to find neighbors for RCMG. Defaults to 1.
            min_size (int, optional): The minimum number of pixels in a region. Defaults to 20.
            n_proc (int, optional): the number of processes for Pool for calculating gradient. Defaults to 8.
        """
        super().__init__(data)
        self.set_markers(markers)
        self.set_connectivity(connectivity)
        self.set_offset(offset)
        self.set_mask(mask)
        self.set_compactness(compactness)
        self.set_watershed_line(watershed_line)
        self.set_dist_func(dist_func)
        self.set_window(window)
        self.set_min_size(min_size)
        self.set_n_proc(n_proc)

    def set_markers(self, markers):
        self.markers = markers
        self._change = True

    def set_connectivity(self, connectivity):
        self.connectivity = connectivity
        self._change = True

    def set_offset(self, offset):
        self.offset = offset
        self._change = True

    def set_mask(self, mask):
        self.mask = mask
        self._change = True

    def set_compactness(self, compactness):
        self.compactness = compactness
        self._change = True

    def set_watershed_line(self, watershed_line):
        self.watershed_line = watershed_line
        self._change = True

    def set_dist_func(self, dist_func):
        self.dist_func = dist_func
        self._change = True

    def set_window(self, window):
        self.window = window
        self._change = True

    def set_min_size(self, min_size):
        self.min_size = min_size
        self._change = True

    def set_n_proc(self, n_proc):
        self.n_proc = n_proc
        self._change = True

    def _rcmg(self, pixels, r=1):
        """Robust Color Morphological Gradient of set of pixels.

        For a set of pixels, it finds the pair with the largest distance and removes those 2 pixels.
        Continues removing pairs of pixels r times.
        Finally returns the remaining largest distance.

        Args:
            pixels (ndarray): an array with a set of pixels (from _get_neighbors).
            r (int, optional): Number of max pairs to remove. Defaults to 1.

        Returns:
            float: returns the RCMG for that set of pixels.
        """
        dists = pairwise_distances(pixels, metric=self.dist_func)
        for i in range(r):
            row, col = np.where(dists == dists.max())
            dists = np.delete(dists, [row[0], col[0]], 0)
            dists = np.delete(dists, [row[0], col[0]], 1)
        return dists.max()

    def _get_neighbors(self, coord, window, img):
        """Gets the neighbors around a pixel

        Given a pixel in the img at (r, c), this returns a list of the neighbors
        of (r,c) in a window (where 1 makes a 3x3 window, 2 makes a 5x5 window).

        Args:
            r (int): the row of the pixel of interest
            c (int): the column of the pixel of interest
            window (int): size of window to get neighbors
            img (ndarray): 3D image where pixel is from.

        Returns:
            ndarray: the set of neighboring pixels.
        """
        r, c = coord[0], coord[1]
        width, height = img.shape[1], img.shape[0]
        ri, ui = max(c - window, 0), max(r - window, 0)
        di, li = min(r + window + 1, height), min(c + window + 1, width)
        res = img[ui:di, ri:li, :]
        return res.reshape(-1, res.shape[-1])

    def _gradient_wrapper(self, coord, window, img):
        pixels = self._get_neighbors(coord, window, img)
        return self._rcmg(pixels), coord

    def _gradient(self, img, window=1, n_proc=16):
        """Calculates the gradient of the image using RCMG.

        Uses the RCMG to calculate the gradient of the img to pass to the watershed algorithm.

        Args:
            img (ndarray): 3D image to segment
            window (int, optional): size of window for calculating RCMG. Defaults to 1.

        Returns:
            ndarray: 2d array of the gradients for each pixel.
        """
        kwargs = {'window': window,
                  'img': img}
        with Pool(n_proc) as p:
            res = p.map(partial(self._gradient_wrapper, **kwargs),
                        np.ndindex((img.shape[0], img.shape[1])))
            p.close()
            p.join()
            res = list(res)
        grad = np.zeros((img.shape[0], img.shape[1]))
        for r in res:
            x, y = r[1]
            grad[x, y] = r[0]
        return grad

    def segment(self, img=None):
        """Segments the image using the watershed algorithm.

        Args:
            img (ndarray, optional): 3D image to segment. Defaults to None.

        Returns:
            ndarray: 2D array with labeled segments.
        """
        img = super().segment(img)
        self.grad = self._gradient(img, self.window, self.n_proc)
        seg = segmentation.watershed(self.grad, markers=self.markers, connectivity=self.connectivity,
                                     offset=self.offset, mask=self.mask,
                                     compactness=self.compactness,
                                     watershed_line=self.watershed_line)

        return _remove_singletons(seg, self.min_size)


class Slurm(Segmenter):
    """Slic Uncertain Region Merging

    This segmentation algorithm is based on first finding an oversegmentation,
    (default is to use Slic, but you can specify any segmentation algorithm, or pass in a segmentation).
    It then merges regions depending on the merge_order you specify.
    There are two modes for merging, cautious merging, and free merging.
    When in free merging, a region will consider all of its neighbors, and
    merge with the neighbor with the least decrease in quality score.
    When in cautious merging, a region will consider all of it's neighbors, and
    merge with the neighbor with the smallest decrease in quality score up to
    some threshold, so it wont merge with a region if it will drastically decrease the quality score.
    The threshold to determine the merging mode is defined by the merge order and high_percentile.
    The merge order can be smallest, lowest, or mixed.
    If smallest, then it will merge smallest regions first, and apply free merging to the smallest regions only.
    Smallest regions is defined as regions whose size is less than 100 - high_percentile of all region sizes.
    So setting high_percentile to 80 means the smallest 20% of regions will be free merged.
    If merge order is lowest, it will merge regions going from lowest SUQ to highest SUQ.
    It will apply free merging only to regions whose SUQ is above the high_percentile value.
    scale_r_with_iters will increase the r value for SUQ with each iter.
    max_per_dec speficies the maximum percent decrease in Q score allowed in cautious/constrained merging.
    Regions will only merge if the new quality score is at worst max_per_dec lower than the original Q score.
    Each iteration's segmentation and SUQ scores are stored in self.segs and self.uqs.
    segs[0] will be the original segmentation, and segs[-1] will be the final merged segmentation.
    Similarly for uqs. Note that uqs[0] is a tuple with the SUQ as a list first, then SUQ in a dict second (when using suq=suqsES).
    """

    def __str__(self) -> str:
        return 'SlURM'

    def __init__(self, img, min_size=20, high_percentile=65, max_per_dec=.15, iters=6, uq_args={'n': 10, 'p': .65, 'r': 2, 'dist_func': dist.cos},
                 seger_args={}, seger=Slic, suq=suqsES, n_thread=4, merge_order='lowest', seg=None,
                 scale_r_with_iters=False):
        """Sets parameters for the algorithm

        The min_size is used to specify n_segments for the slic algorithm (or watershed algorithm).
        min_size says (approximatley) what is the minimum number of pixels for each segment
        in the original segmentation. It finds n_segments as (width * height) / min_size, where
        width and height are of the image.
        The high_percentile defines a cutoff between contrained and un-constrained merging.
        See class docstring for better description.
        A higher high_percentile will result in less un-constrained merging (more constraind merging).
        max_per_dec is how much worse can merged regions get (as a percentage of the original Q score).
        A higher max_per_dec should result in more merging for regions doing constrained/cautious merging.
        The number of processes spun up is based on n_thread, but optimized based on computation time.
        For small tasks, .5 * n_thread processes are started, but for long tasks (like uq)
        1.5 * n_thread processes are started.
        This can be overridden by specifying n_thread in uq_args.

        Args:
            img (dict or ndarray): a dictionary with at least an 'img' key with 3D hyperspectral image, or just the 3D array.
            min_size (int, optional): See above. Defaults to 9.
            high_percentile (int, optional): see above. Defaults to 65.
            max_per_dec (float, optional): see above. Defaults to .15.
            iters (int, optional): total number of iterations to run. Defaults to 6.
            uq_args (dict, optional): args to pass to the SUQ score class. Defaults to {'n': 10, 'p': .65, 'r': 2, 'dist_func':dist.cos}.
            seger_args (dict, optional): args to pass to the segmentation algorithm. Defaults to {}.
            seger (Segmenter, optional): what segmenter to make initial segmentation. Defaults to Slic.
            suq (UQMetric, optional): what SUQ score to use. Defaults to suqsES.
            n_thread (int, optional): number of processes to use in calculations. Defaults to 4.
            merge_order (str, optional): can be 'lowest', 'smallest', 'mixed'. Defaults to 'lowest'.
            seg (ndarray, optional): a pre-made segmentation. Defaults to None.
            scale_r_with_iters (bool, optional): whether to increase r for SUQ score. Defaults to False.

        Raises:
            RuntimeError: makes sure the merge order is correct
        """
        super().__init__(img)
        if self.img is not None:
            self.set_min_size(min_size)
            self.set_seger(seger)
            self.set_seger_args(seger_args)
            self.set_n_thread(n_thread)
            self.set_suq_c(suq)
            self.set_uq_args(uq_args)
            self.set_high_percentile(high_percentile)
            self.set_max_per_dec(max_per_dec)
            self.set_iters(iters)
            self.set_merge_order(merge_order)
            self.set_scale_r_with_iters(scale_r_with_iters)
            self.set_seg(seg)
            self.uqs = []

    def set_min_size(self, min_size):
        self.min_size = min_size
        self._change = True

    def set_seger_args(self, seger_args):
        self.seger_args = seger_args
        if self.seger_name == 'Slic':
            if 'n_segments' not in seger_args.keys():
                wid, hei = self.img.shape[1], self.img.shape[0]
                self.seger_args['n_segments'] = int(
                    (wid * hei) / self.min_size)
        elif self.seger_name == 'Watershed':
            if 'markers' not in seger_args.keys():
                wid, hei = self.img.shape[1], self.img.shape[0]
                self.seger_args['markers'] = int(
                    (wid * hei) / self.min_size)
        self._change = True

    def set_uq_args(self, uq_args):
        if hasattr(self, 'uq_args'):
            self.uq_args = {**self.uq_args, **uq_args}
        else:
            self.uq_args = uq_args
            if 'dist_func' not in self.uq_args.keys():
                self.uq_args['dist_func'] = dist.cos
            if 'n' not in self.uq_args.keys():
                self.uq_args['n'] = 10
            if 'p' not in self.uq_args.keys():
                self.uq_args['p'] = .65
            if 'r' not in self.uq_args.keys():
                self.uq_args['r'] = 2
            if 'n_thread' not in self.uq_args.keys():
                self.uq_args['n_thread'] = int(1.5 * self.n_thread)
        # if hasattr(self, 'suq'):
        self.suq = self.suq_c(self.img, None, **self.uq_args)
        self._change = True

    def set_seger(self, seger):
        self.seger = seger
        self.seger_name = str(seger(None))
        self._change = True

    def set_n_thread(self, n_thread):
        self.n_thread = n_thread
        # self.uq_args['n_thread'] = int(1.5 * n_thread)
        self._change = True

    def set_high_percentile(self, high_percentile):
        self.high_percentile = high_percentile
        self._change = True

    def set_max_per_dec(self, max_per_dec):
        self.max_per_dec = max_per_dec
        self._change = True

    def set_iters(self, iters):
        self.iters = iters
        self._change = True

    def set_merge_order(self, merge_order):
        merge_order = merge_order.lower()
        if merge_order != 'smallest' and merge_order != 'lowest' and merge_order != 'mixed':
            raise RuntimeError(
                f'Need merge_order to be \'smallest\' or \'lowest\' or \'mixed\'.')
        else:
            self.merge_order = merge_order.lower()
        self._change = True

    def set_suq_c(self, suq_c):
        self.suq_c = suq_c
        self._change = True

    def set_scale_r_with_iters(self, scale_r_with_iters):
        self.scale_r_with_iters = scale_r_with_iters
        if self.scale_r_with_iters:
            self.rs = [i+1 for i in range(self.iters)]
        else:
            self.rs = [self.uq_args['r'] for i in range(self.iters)]
        self._change = True

    def set_seg(self, seg):
        self.segs = []
        if seg is not None:
            self.segs.append(seg)
        self._change = True

    def _calc_neigh_q_scores(self, regs):
        """Finds q score for two regions merged

        if regs is a tuple, then it finds the q score
        as if the regions were merged.
        If it is not a tuple then it returns None

        Args:
            regs (tuple): regions to find merged q score

        Returns:
            float or None: returns q score, or none
        """
        if isinstance(regs, tuple):
            return regs, self.suq.get_q_score_reg(*regs)
        else:
            return None

    def _calc_neigh_change_q_score(self, reg, neighbors):
        """Finds how much Q scores if merged with neighbors

        neighbors should have neighbors[reg] be a list of reg's neighbors labels.
        And neighbors[(reg, n)] should have q_scores for pairs of regions.
        Returns 1. reg
        2. the smallest percent decrease in quality score
        3. the neighbor's label which has the smallest decrease in quality score.
        4. list of all neighboring percent change in quality score

        Args:
            reg (int): the region of interest
            neighbors (dict): contains q scores for neighbors

        Returns:
            4-tuple: see above
        """
        qs_og = self.suq.get_q_score_reg(reg)
        qs = []
        for n in neighbors[reg]:
            if reg < n:
                p = (reg, n)
            else:
                p = (n, reg)
            qs_n = neighbors[p]
            qs.append((n, (qs_og - qs_n) / qs_og))
        qs = sorted(qs, key=lambda x: x[1])
        return reg, qs[0][1], qs[0][0], qs

    def _find_neighbors(self, l, seg):
        """finds region l's neighbors

        Finds all regions that neighbor region l.
        Returns 1. l, the label for the region of interest
        2. list of all neighbor label pairs (sorted smallest label first)
        3. list of all neighboring labels.

        Args:
            l (int): region of interest
            seg (ndarray): segmentation matrix

        Returns:
            3-tuple: see above.
        """
        xs, ys = np.where(segmentation.find_boundaries(seg == l, mode='outer'))
        neighs = np.unique(seg[xs, ys])
        ps = []
        for n in neighs:
            if l < n:
                p = (l, n)
            else:
                p = (n, l)
            ps.append(p)
        return l, ps, neighs

    def _find_all_neighbors(self, seg):
        """finds all neighboring region pairs

        Uses multiprocessing pool.
        return a dict where there are 2 types of keys.
        int keys correspond to each region label, the value
        is a list of that regions neighbors.
        tuple keys (sorted (smaller, larger)) are to track
        all unique pairs of neighbors, the value is None.

        Args:
            seg (ndarray): segmentation matrix

        Returns:
            dict: dict with int and tuple keys
        """
        neighbors = {}
        with Pool(int(self.n_thread / 2)) as p:
            args = {'seg': seg}
            res = p.map(partial(self._find_neighbors, **args), np.unique(seg))
            p.close()
            p.join()
            res = list(res)
        for l, ps, neighs in res:
            neighbors[l] = neighs
            for p in ps:
                neighbors[p] = None
        return neighbors

    def _find_cutoff(self, uq_l, uq_d, seg, iter, merge_order):
        """Finds the cutoff threshold according to merge_order

        Allowable merge_orders are smallest, lowest, and mixed.
        if smallest, it will sort regions by their size, and find
        the threshold according to 100-high_percentile of size.
        If lowest, it will sort regions lowest UQ to highest, and find
        the cutoff according to high_percentile of suq scores.
        If mixed, if the iter is even, then it will do a lowest merge,
        if the iter is odd it will do a smallest merge.
        Returns 1. uq_d, dict of suq scores,
        2. the cutoff value
        3. sizes of the regions as dict,
        4. merge_order (for facilitating mixed merging).

        Args:
            uq_l (list): list of all uq scores
            uq_d (dict): dict with region, suq score pairs
            seg (ndarray): segmentation matrix
            iter (int): current iteration
            merge_order (str): the merge order

        Returns:
            4-tuple: see above
        """
        if merge_order == 'lowest':
            # Sorts lowest to highest UQ
            uq_d = dict(sorted(uq_d.items(), key=lambda item: item[1][0]))
            cutoff = np.percentile(
                uq_l, self.high_percentile, method='averaged_inverted_cdf')
            sizes = None
        elif merge_order == 'smallest':
            # Sorts by smallest region
            sizes = {}
            for k in uq_d.keys():
                sizes[k] = np.sum(seg == k)
            uq_d = dict(
                sorted(uq_d.items(), key=lambda item: sizes[item[0]]))
            cutoff = np.percentile(
                [v for k, v in sizes.items()], 100 - self.high_percentile, method='averaged_inverted_cdf')
        elif merge_order == 'mixed':
            if iter % 2 == 0:
                return self._find_cutoff(uq_l, uq_d, seg, iter, 'lowest')
            else:
                return self._find_cutoff(uq_l, uq_d, seg, iter, 'smallest')
        return uq_d, cutoff, sizes, merge_order

    def _merge_regions(self, seg, merge_info, cutoff, uq_d, sizes, merge_order):
        """logic for how to merge regions

        the merge_info should come from finding all the neighbors,
        finding all the change in q scores for those neighbors.

        Args:
            seg (ndarray): segmentation matrix.
            merge_info (dict): dict with all the changes in q score.
            cutoff (float): the cutoff to determine to free or constrained merge.
            uq_d (dict): dict with all the SUQ scores.
            sizes (dict): dict with the size of all regions.
            merge_order (str): how to merge, smallest or lowest.

        Returns:
            ndarray: new merged segmentation matrix (labels will not be sequential)
        """
        for r in merge_info:
            cur_key = r[0]
            if not np.any(seg == cur_key):
                continue
            if merge_order == 'lowest':
                val = uq_d[cur_key][0]
                direction = '<='
            elif merge_order == 'smallest':
                val = sizes[cur_key]
                direction = '>='
            if eval(f'val {direction} cutoff'):
                best_score = r[1]
                if best_score <= self.max_per_dec:
                    m_key = r[2]
                    seg = np.where(seg == m_key, cur_key, seg)
            else:
                m_key = r[2]
                seg = np.where(seg == m_key, cur_key, seg)
        return seg

    def segment(self, img=None):
        """runs the slurm segmentation algorithm

        Args:
            img (dict or ndarray, optional): a dictionary with at least an 'img' key with 3D hyperspectral image, or just the 3D array. Defaults to None.

        Returns:
            ndarray: final segmentation matrix
        """
        img = super().segment(img)
        if len(self.segs) == 0:
            seg = self.seger(img, **self.seger_args).get_segmentation()
            self.segs.append(seg)
        else:
            seg = self.segs[0]
        for it in tqdm(range(self.iters)):
            if len(np.unique(seg)) <= 2:
                warnings.warn(f'Only 2 segments left, stopping early.')
                break
            # get original segmentation and SUQs
            self.suq.set_info(seg.copy())
            self.suq.set_r(self.rs[it])

            uq_l, uq_d = self.suq.get_uq()
            uq_d, cutoff, sizes, merge_order = self._find_cutoff(
                uq_l, uq_d, seg, it, self.merge_order)
            self.uqs.append((uq_l, uq_d))

            # Find all neighbor scores
            neighbors = self._find_all_neighbors(seg)
            res = []
            with Pool(self.n_thread) as p:
                res = p.map(self._calc_neigh_q_scores, neighbors.keys())
                p.close()
                p.join()
                res = list(res)
            # updates the neighbors scores
            for r in res:
                if r is not None:
                    neighbors[r[0]] = r[1]
            # Calcultes the change in scores
            with Pool(int(self.n_thread / 2)) as p:
                args = {'neighbors': neighbors}
                res = p.map(
                    partial(self._calc_neigh_change_q_score, **args), np.unique(seg))
                p.close()
                p.join()
                res = list(res)
            merge_info = res

            # start merging regions
            seg = self._merge_regions(
                seg, merge_info, cutoff, uq_d, sizes, merge_order)
            seg = label_clusters(seg)
            self.segs.append(seg)

        self.suq.set_info(self.segs[-1])
        uq_l, uq_d = self.suq.get_uq()
        self.uqs.append((uq_l, uq_d))
        return seg


class Hierarchical(Segmenter):
    def __str__(self) -> str:
        return 'Hierarchical Clustering'

    def __init__(self, img, n_segments=100, connectivity=50, connect_args={}, 
                 cluster_args={'affinity':'cosine', 'linkage':'average'}, min_size=20):
        super().__init__(img)
        self.set_n_segments(n_segments)
        self.set_connectivity(connectivity)
        self.set_connect_args(connect_args)
        self.set_cluster_args(cluster_args)
        self.set_min_size(min_size)

    def set_n_segments(self, n_segments):
        self.n_segments = n_segments
        self._change = True

    def set_connectivity(self, connectivity):
        self.connectivity = connectivity
        self._change = True

    def set_connect_args(self, connect_args):
        if hasattr(self, 'connect_args'):
            self.connect_args = {**self.connect_args, **connect_args}
        else:
            self.connect_args = connect_args
        self._change = True

    def set_cluster_args(self, cluster_args):
        if hasattr(self, 'cluster_args'):
            self.cluster_args = {**self.cluster_args, **cluster_args}
        else:
            self.cluster_args = cluster_args
        self._change = True

    def set_min_size(self, min_size):
        self.min_size = min_size
        self._change = True

    def _create_connectivity(self, img, connectivity):
        con_data = np.empty_like(img[:, :, 0:2])
        for i in range(con_data.shape[0]):
            for j in range(con_data.shape[1]):
                con_data[i, j, 0] = i
                con_data[i, j, 1] = j
        con_data = con_data.reshape((con_data.shape[0] * con_data.shape[1], con_data.shape[2]))
        con_data = kneighbors_graph(con_data, n_neighbors=connectivity, include_self=False, **self.connect_args)
        return con_data

    def segment(self, img=None):
        img = super().segment(img)

        if isinstance(self.connectivity, int):
            connectivity_info = self._create_connectivity(img, self.connectivity)
        else:
            connectivity_info = self.connectivity

        img_data = self.img.reshape(self.img.shape[0] * self.img.shape[1], self.img.shape[2])
        
        clusters = AgglomerativeClustering(n_clusters=self.n_segments, connectivity=connectivity_info,
                                           **self.cluster_args).fit(img_data)
        self.clusters = clusters

        seg = np.reshape(clusters.labels_, (self.img.shape[0], self.img.shape[1]))
        return _remove_singletons(seg, self.min_size, relabel=False)

    def get_segmentation(self):
        return super().get_segmentation(False)

