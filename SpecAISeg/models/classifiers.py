from functools import partial
from re import L

import numpy as np
from multiprocess.pool import Pool
from scipy.linalg import sqrtm
from skimage.segmentation import find_boundaries
from spectral.algorithms import mean_cov

from ..utils import get_pixels, get_roi
from ..whitening import roi_segment_stats, whiten_img
from ._classifiers import trainedmodels


class TrainedModel():

    def __init__(self, file_path, roi, ds, img_key='img_raw', rgb_key='img_rgb', seg=None, whiten=True, whiten_args={}, mask=False, load_model_args={}):
        """Load a trained model for inference on regions of interest

        This will load a model from the models directory, and will load
        an ROI from a json file, and make predictions for those regions of interest.
        The image key is the key used on the ds to get the hyperspectral image,
        similarly the rgb_key is the key for the rgb image.
        Seg is a segmentation matrix if you want to use local whitening, but haven't computed
        the whitened image first. The whiten_args will be passed to whiten_img.
        You can also pass the tule result from whiten_img for the whiten arg,
        or you can just pass in the whitened image array for the whiten argument.
        Be sure to include a library_clusters.json as cluter_file arg in load_model_args.

        Args:
            file_path (str): file path for the model to load.
            roi (str or dict): path to roi, or already loaded roi dict.
            ds (dict): dataset with an image to use for classification, and an rgb image.
            img_key (str, optional): key to get the classification image from ds. Defaults to 'img_raw'.
            rgb_key (str, optional): key to get the rgb image from ds. Defaults to 'img_rgb'.
            seg (ndarray, optional): segmentation array, used for local whitening. Defaults to None.
            whiten (bool, optional): whether to whiten the data, tuple from whiten_img, or just a whitened ndarray. Defaults to True.
            whiten_args (dict, optional): potential args to pass to the whiten_img function. Defaults to {}.
            load_model_args (dict, optional): arguments for load_torch_model. Defaults to {}.
        """
        self.model = trainedmodels.load_torch_model(file_path, **load_model_args)


        self.file_path = file_path
        if isinstance(roi, str):
            self.roi_dict = get_roi(roi, ds[img_key], ds[rgb_key])
        elif isinstance(roi, dict):
            self.roi_dict = roi
        else:
            raise RuntimeWarning('roi should be a string or dict.')
        self.ds = ds
        self.img_key = img_key
        self.rgb_key = rgb_key
        self.seg = seg
        self.whiten = whiten
        self.whiten_args = whiten_args
        self.mask = mask

        if isinstance(whiten, bool):
            self.roi_dict_pred = self._create_roi_pred_dict(self.roi_dict, ds[img_key], self.seg, True)
        elif isinstance(whiten, tuple):
            self.img, self._means, self._isr_covs = whiten
            self.roi_dict_pred = self._create_roi_pred_dict(self.roi_dict, self.img, self.seg, False)
        elif isinstance(whiten, np.ndarray):
            self.img = whiten
            self.roi_dict_pred = self._create_roi_pred_dict(self.roi_dict, self.img, self.seg, False)
        else:
            self.img = self.ds[self.img_key]
            self.roi_dict_pred = self._create_roi_pred_dict(self.roi_dict, self.img, self.seg, False)
        self.img_rgb = self.ds[self.rgb_key]
        

    def _create_roi_pred_dict(self, roi_dict, img, seg, whiten):
        """Converts roi_dict to the dict the model likes

        Essentially just returns a dict with the same keys,
        but the values are the mean spectra from the image
        for the pixels in the regions of interest.

        Args:
            roi_dict (dict): result from get_roi, should have 'IMG_MASK' key for each roi key.
            img (ndarray): image array to get pixels from (can be the whitened image).

        Returns:
            dict: same first level keys as roi_dict, mean spectra for value.
        """
        if seg is None:
            seg = np.ones(img.shape[0:2])
        roi_dict_pred = {}
        for k in self.roi_dict.keys():
            if whiten:
                regs, _, __, ___ = roi_segment_stats(k, roi_dict, seg)
                mask = None
                if self.mask:
                    mask = roi_dict[k]['IMG_MASK']
                img_white, _, __ = whiten_img(img, seg, regs, roi_mask=mask, **self.whiten_args)
                pixels = get_pixels(True, img_white, roi_dict[k]['IMG_MASK'])
            else:
                pixels = get_pixels(True, img, roi_dict[k]['IMG_MASK'])
            roi_dict_pred[k] = np.mean(pixels, axis=0)
        return roi_dict_pred

    def print_results(self, result, n=5):
        """Given results from predict_roi, print the top n nicely

        Args:
            result (dict): result from predict_roi function
            n (int, optional): number of entries to print out. Defaults to 5.
        """
        for ikey in result:
            num_substances = len(result[ikey]['scores'])
            print('\nNumber of substances for ROI {} in probability scores: {}'.format(ikey, num_substances))
            print('  ROI ID, Substance Name, Score')
            for idx in range(n):
                print('\t', ikey, result[ikey]['names'][idx], '\t', result[ikey]['scores'][idx])

    def predict_roi(self, roi_dict=None, img=None, n=None):
        """given an roi_dict, make predictions

        Args:
            roi_dict (dict, optional): result from get_roi. Defaults to None.
            img (ndarry, optional): image array to use. Defaults to None.
            n (int, optional): print top n predictions, if None it wont print results. Defaults to 5.

        Returns:
            dict: results from predict_dict
        """
        if roi_dict is None:
            roi_dict_pred = self.roi_dict_pred
        else:
            roi_dict_pred = self._create_roi_pred_dict(roi_dict, img)
        
        results = self.model.predict_dict(roi_dict_pred)
        if n is not None:
            self.print_results(results, n)
        return results
        