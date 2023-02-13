import warnings
from abc import ABC, abstractmethod
from functools import partial
from multiprocessing.pool import Pool

# import graph_tool as gt
import numpy as np
# from graph_tool.topology import label_components
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
    seg = label_clusters(mat)
    counts = np.bincount(seg.flatten())
    return seg, counts
    # g = gt.Graph(directed=False)
    # verts = np.empty_like(mat, dtype=gt.Vertex)
    # for r in range(mat.shape[0]):
    #     for c in range(mat.shape[1]):
    #         v = g.add_vertex()
    #         verts[r, c] = v

    # for r in range(mat.shape[0]):
    #     for c in range(mat.shape[1]):
    #         lab = mat[r, c]
    #         lu = r - 1 >= 0
    #         ld = r + 1 < verts.shape[0]
    #         ll = c - 1 >= 0
    #         lr = c + 1 < verts.shape[1]

    #         if lu and mat[r - 1, c] == lab:
    #             g.add_edge(verts[r - 1, c], verts[r, c])
    #         if ld and mat[r + 1, c] == lab:
    #             g.add_edge(verts[r + 1, c], verts[r, c])
    #         if ll and mat[r, c - 1] == lab:
    #             g.add_edge(verts[r, c - 1], verts[r, c])
    #         if lr and mat[r, c + 1] == lab:
    #             g.add_edge(verts[r, c + 1], verts[r, c])

    # comp, counts = label_components(g, directed=False)
    # i = 0
    # res = np.empty_like(mat)
    # for r in range(mat.shape[0]):
    #     for c in range(mat.shape[1]):
    #         res[r, c] = comp[i]
    #         i += 1
    # return res.astype(int), counts


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

