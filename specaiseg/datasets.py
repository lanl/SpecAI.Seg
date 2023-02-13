import ast
import os
import warnings
from abc import ABC, abstractmethod
from pathlib import Path
from urllib.request import urlretrieve
import pkg_resources

import numpy as np
import pandas as pd
import torch
import torch.utils.data

from .models.segmenters import label_clusters
from .utils import TqdmUpTo, false_grey_img, open_file


def create_dataset(dataset_name, **kwargs):
    """
    Combines getting/downloading data_raw, and making it into a torch Dataset

    Takes a dataset name and other arguments to create a Torch Dataset
    The kwargs are passed to get_data and the specified Dataset class.
    Use set_type (str) as patches, disjoint to pick which type of dataset you want.

    Parameters
    ----------
        dataset_name : str
            The name of the dataset you want to download
        kwargs:
            Passes arguments to get_data and specified Dataset function
    """
    target_folder = kwargs.get('target_folder', './Data')
    datasets = kwargs.get('datasets', None)

    data = get_data(dataset_name, target_folder, datasets)

    set_type = kwargs.get('set_type', None).lower()

    if set_type == 'disjoint':
        dataset = DatasetDisjoint(data, **kwargs)
    else:
        dataset = DatasetPatches(data, **kwargs)

    return dataset


def get_custom_data(file, clip_p=0.0025, load_function=open_file, load_function_args={}, rgb_channels=[0, 1, 2], rgb_wavelengths=[11.3, 10.2, 8.6],
                    name=None):
    """Creates a dataset from a given file

    This will return a similar dictionary as get_data, but slightly different.
    For one this won't have ground truth associated with it, unless the
    load function returns it.
    The load_function defaults to open_file, which can handle most cases.
    But if your data is not supported by load_function, then you can pass in your
    own load_function (note that the first argument should be the file path string),
    and you can pass arguments to it using the load_function_args dict.
    The load_function should either just return the image array, or a dict with a key 'img'
    for the image, and 'wavelength' if the wavelength info is included.
    The rgb channels specify what channels to use for the rgb image.
    If the load_function return wavelengths then these wavelengths will be used in the
    utils.false_grey_img function.
    Like get_data, the image will be clipped, and then scaled to be between 0 and 1.
    Though the raw and clipped versions will also be returned.

    Args:
        file (str): the path of the file you want to load.
        clip_p (float, optional): clips the top and bottom values according to this percent. Defaults to 0.0025.
        load_function (function, optional): a function to load the image. Defaults to open_file.
        load_function_args (dict, optional): arguments to pass to load function. Defaults to {}.
        rgb_channels (list, optional): what channels to act as RGB channels. Defaults to [0, 1, 2].
        rgb_wavelengths (list, optional): what wavelengths to act as RGB channels. Defaults to [11.3, 10.2, 8.6].
        name (str, optional): name given to the dataset, if none given filename will be used. Defaults to None.

    Returns:
        dict: dictionary with all the information, img (the clipped and standardized image),
        img_raw (the original image returned from load_function), img_clipped (the image with values clipped),
        img_rgb (a false color rgb image), img_gray (gray scale image, may be None),
        loaded (the results returned from load_function), wavelengths (wavelength info, may be None),
        name (the name given, or the filename).
    """
    data = load_function(file, **load_function_args)
    wavelengths = None
    if isinstance(data, dict):
        img = data['img']
        if 'wavelength' in data.keys():
            wavelengths = data['wavelength']
    else:
        img = data

    if wavelengths is not None:
        try:
            img_gray, img_rgb = false_grey_img(img, wavelengths, rgb_wavelengths)
        except:
            img_rgb = img[:, :, rgb_channels]
            img_gray = None
    else:
        img_rgb = img[:, :, rgb_channels]
        img_gray = None

    img_raw = img.copy()
    # Normalizes the image
    img = np.asarray(img, dtype="float32")
    idx = int(clip_p * img.size)
    max = np.partition(img.flatten(), -idx)[-idx]
    min = np.partition(img.flatten(), idx)[idx]
    img = np.clip(img, min, max)
    img_clipped = img.copy()
    img = (img - np.min(img)) / (np.max(img) - np.min(img))

    if name is None:
        name = Path(file).stem

    return {'img': img,
            'img_raw': img_raw,
            'img_clipped': img_clipped,
            'img_rgb': img_rgb,
            'img_gray': img_gray,
            'loaded': data,
            'wavelengths': wavelengths,
            'name': name}


def get_data(dataset_name, target_folder='./Data', datasets=None, clip_p=0.0025):
    """
    Load or download one of the predefined datasets.

    The datasets_config shows the datasets that can be easily loaded/downloaded.
    They are Botswana, IndianPines, KSC, PaviaC, PaviaU, and Salinas.
    dataset_name can be any of the above datasets, target_folder is where to download to or load from,
    and datasets should be left None in order to use the default parameters.

    Parameters
    ----------
    dataset_name : str
        Name of dataset to load, such as Botswana, IndianPines, KSC, PaviaC, PaviaU, or Salinas.
    target_folder : str
        The directory where you want to download the data_raw, or where to read the data_raw from.
    datasets: dict
        Dictionary with at least urls, img, img_key, gt, gt_key, rgb_bands, label_values, ignored_labels
    clip_p: float
        clip the top and bottom clip_p percent of values when standardizing.

    Returns
    -------
    dict
        All the needed data_raw, namely img (image clipped and standardized), img_raw (original image),
        img_clipped (clipped but unstandardized), gt (ground truth/labels),
        label_values (name corresponding to label (i.e. 1=water, etc),
        seg (ground truch segmentation, not labeled according to label_values),
        img_rgb (the image with just the rgb_channels),
        ignored_labels (background or labels to ignore), rgb_bands (which HSI bands to use for RGB image),
        name (dataset_name)
    """

    if datasets is None:
        # Reads in datasets.csv file which has the urls to download standard, 1 image datasets
        this_dir, this_filename = os.path.split(__file__)
        data_file = pkg_resources.resource_filename('specaiseg', 'data_raw/datasets.csv')
        datasets = pd.read_csv(data_file, index_col=0).to_dict()
        # datasets = pd.read_csv(os.path.join(
        #     this_dir, 'data_raw/datasets.csv'), index_col=0).to_dict()
        # Converts urls etc to actually be lists
        for name in datasets:
            for var in datasets[name]:
                if var in ['urls', 'rgb_bands', 'label_values', 'ignored_labels']:
                    datasets[name][var] = ast.literal_eval(datasets[name][var])
    else:
        # Makes sure custom config has necessary keys
        required = {'urls', 'img', 'img_key', 'gt', 'gt_key',
                    'rgb_bands', 'label_values', 'ignored_labels'}
        for k in datasets:
            if not required.issubset(datasets[k].keys()):
                raise ValueError(f'datasets must have the keys: {required}')

    # Makes sure the dataset is in the available datasets
    if dataset_name.lower() not in [k.lower() for k in datasets.keys()]:
        raise ValueError(f'{dataset_name} dataset is unknown.')

    # Allows for lowercase spelling when passing in dataset_name, but keeps proper spelling for files and such
    dataset_name = [k for k in datasets.keys() if k.lower() ==
                    dataset_name.lower()][0]
    dataset = datasets[dataset_name]

    # The folder to download or read in the data_raw
    folder = os.path.join(target_folder, dataset.get('folder', dataset_name))

    if dataset.get('download', True):
        # Download the dataset if it is not present
        if not os.path.isdir(folder):
            os.makedirs(folder)
        for url in dataset['urls']:
            # download the files
            filename = url.split('/')[-1]
            if not os.path.exists(os.path.join(folder, filename)):
                with TqdmUpTo(
                        unit='B',
                        unit_scale=True,
                        miniters=1,
                        desc=f'Downloading {filename}',
                ) as t:
                    urlretrieve(url, filename=os.path.join(
                        folder, filename), reporthook=t.update_to)
    elif not os.path.isdir(folder):
        warnings.warn(f'{dataset_name} is not downloadable.')

    # Loads in the image and gt
    img = open_file(os.path.join(folder, dataset['img']))[dataset['img_key']]
    gt = open_file(os.path.join(folder, dataset['gt']))[dataset['gt_key']]
    rgb_bands = dataset['rgb_bands']
    label_values = dataset['label_values']
    ignored_labels = dataset['ignored_labels']

    # Filter NaN out
    nan_mask = np.isnan(img.sum(axis=-1))
    if np.count_nonzero(nan_mask) > 0:
        warnings.warn("NaN have been found in the data_raw. It is preferable to remove them beforehand. Learning "
                      "on NaN data_raw is disabled. ")
    img[nan_mask] = 0
    gt[nan_mask] = 0
    ignored_labels.append(0)
    ignored_labels = list(set(ignored_labels))

    img_raw = img.copy()

    # Normalizes the image
    img = np.asarray(img, dtype="float32")
    idx = int(clip_p * img.size)
    max = np.partition(img.flatten(), -idx)[-idx]
    min = np.partition(img.flatten(), idx)[idx]
    img = np.clip(img, min, max)
    img_clipped = img.copy()
    img = (img - np.min(img)) / (np.max(img) - np.min(img))

    return {'img': img,
            'img_raw': img_raw,
            'img_clipped': img_clipped,
            'gt': gt,
            'seg': label_clusters(gt),
            'label_values': label_values,
            'ignored_labels': ignored_labels,
            'rgb_bands': rgb_bands,
            'img_rgb': img[:, :, rgb_bands],
            'name': dataset_name}


class _DatasetSingleImage(ABC, torch.utils.data.Dataset):
    """
    Private abstraction of a torch Dataset for single images

    This provides the main functions for creating a torch dataset with use of a single image.
    There are three functions which need to be implemented to create a new Dataset.
    This should be the backbone for any Dataset which processes a single image.
    """

    @abstractmethod
    def __additional_parameters(self, **hyperparams):
        """ Allows custom hyper parameters to be set """
        ...

    @abstractmethod
    def __create_indices(self, mask):
        """ Defines what indices are valid """
        pass

    @abstractmethod
    def __get_data(self, idx):
        """ Gets the data and labels at specified index """
        pass

    def __init__(self, data, **hyperparams):
        """
        Constructs all the necessary attributes for the dataset

        The configurable hyper-parameters are:
            patch_size (optional): int, size of the spatial neighbourhood (default 5)
            flip_augmentation (optional): bool, apply vertical and/or vertical flip augmentations (default True)
            flip_p (optional): float, chance of applying flip (default .5)
            radiation_augmentation (optional): bool, apply radiation augmentation (default False)
            radiation_p (optional): float, chance of applying radiation (default .1)
            mixture_augmentation (optional): bool, apply mixture radiation augmentation (default False)
            mixture_p (optional): float, chance of applying mixture (default .2)
            center_pixel (optional): bool, makes label a single pixel (being the center pixel) (default True)
            supervision (optional): str, 'full' train on non-ignored pixels, 'semi' use all pixels except padding, 'all'
             same as semi (default 'all')
            seed (optional): int, seed for random numbers (default 42069)

        Parameters
        ----------
            data : dict
                dictionary with img, gt, name, and ignored_labels (see get_data)
            hyperparams
                Additional hyperparams to configure behavior, see above
        """
        super(_DatasetSingleImage, self).__init__()
        self.img = data['img']
        self.lab = data['gt']
        self.name = data['name']
        self.ignored_labels = data['ignored_labels']

        self.flip_augmentation = hyperparams.get('flip_augmentation', True)
        self.flip_p = hyperparams.get('flip_p', .5)
        self.radiation_augmentation = hyperparams.get(
            'radiation_augmentation', False)
        self.radiation_p = hyperparams.get('radiation_p', .1)
        self.mixture_augmentation = hyperparams.get(
            'mixture_augmentation', False)
        self.mixture_p = hyperparams.get('mixture_p', .2)

        self.patch_size = hyperparams.get('patch_size', 5)
        self.p = self.patch_size // 2
        self.center_pixel = hyperparams.get('center_pixel', True)
        self.supervision = hyperparams.get('supervision', 'All').lower()
        self.seed = hyperparams.get('seed', 42069)
        self.random = np.random.default_rng(self.seed)

        # Process additional hyper parameters if specific to new Dataset
        self.__additional_parameters(**hyperparams)

        # Fully supervised : use all pixels with label not ignored
        if self.supervision == 'full':
            mask = np.ones_like(self.lab)
            for label in self.ignored_labels:
                mask[self.lab == label] = 0
        # Semi-supervised : use all pixels, except padding
        elif self.supervision == 'semi':
            mask = np.ones_like(self.lab)
        else:
            mask = np.ones_like(self.lab)

        # Creates indices according to new Dataset
        self.indices = self.__create_indices(mask)

    def __len__(self):
        """ Returns the total number of patches """
        return len(self.indices)

    def __getitem__(self, idx):
        """ Returns a patch and label for given index """
        patch, label = self.__get_data(idx)

        patch, label = self.__augment_data(patch, label)

        # Copy the data_raw into numpy arrays (PyTorch doesn't like numpy views)
        patch = np.asarray(np.copy(patch).transpose(
            (2, 0, 1)), dtype='float32')
        label = np.asarray(np.copy(label), dtype='int64')

        # Load the data_raw into PyTorch tensors
        patch = torch.from_numpy(patch)
        label = torch.from_numpy(label)
        # Extract the center label if needed
        if self.center_pixel:
            label = label[self.p, self.p]
        # Remove unused dimensions when we work with individual spectra
        elif self.patch_size == 1:
            patch = patch[:, 0, 0]
            label = label[0, 0]

        # Add a fourth dimension for 3D CNN
        if self.patch_size > 1:
            # Make 4D data_raw ((Batch x) Planes x Width x Height)
            patch = patch.unsqueeze(0)
        return patch, label

    def __augment_data(self, patch, label):
        """ Applies flup, radiation and mixture augmentations """
        if self.flip_augmentation:
            patch, label = self.__flip(patch, label)
        if self.radiation_augmentation and self.random.random() < self.radiation_p:
            patch = self.__radiation_noise(patch)
        if self.mixture_augmentation and self.random.random() < self.mixture_p:
            patch = self.__mixture_noise(patch, label)
        return patch, label

    def __flip(self, *arrays):
        """ Randomly flips the arrays horizontally and vertically """
        horizontal = self.random.random() > self.flip_p
        vertical = self.random.random() > self.flip_p
        if horizontal:
            arrays = [np.fliplr(arr) for arr in arrays]
        if vertical:
            arrays = [np.flipud(arr) for arr in arrays]
        return arrays

    def __mixture_noise(self, data, label, beta=1 / 25):
        """ Adds mixture noise to patch """
        alpha1, alpha2 = self.random.uniform(0.01, 1.0, size=2)
        noise = self.random.normal(loc=0.0, scale=1.0, size=data.shape)
        data2 = np.zeros_like(data)
        for idx, value in np.ndenumerate(label):
            if value not in self.ignored_labels:
                l_is = np.nonzero(self.labels == value)[0]
                l_i = self.random.choice(l_is)
                assert self.labels[l_i] == value
                x, y = self.indices[l_i]
                data2[idx] = self.img[x, y]
        return (alpha1 * data + alpha2 * data2) / (alpha1 + alpha2) + beta * noise

    @staticmethod
    def __radiation_noise(data, alpha_range=(0.9, 1.1), beta=1 / 25):
        """ adds radiation noise to patch """
        alpha = np.random.uniform(*alpha_range)
        noise = np.random.normal(loc=0.0, scale=1.0, size=data.shape)
        return alpha * data + beta * noise


class DatasetPatches(_DatasetSingleImage):
    """
    Creates a Torch Dataset for a single image using patches.

    Given data (from get_data, or a dict with img, gt, ignored_labels, and name) it creates as Torch Dataset.
    This can then be used to create a dataset loader.
    Specifically this takes a single image, returns patches of the image for training/testing.
    Note that these patches will overlap (i.e. pixels in training set could be present in testing set).
    See DatasetDisjoint for creating a dataset with patches that don't overlap.
    """

    def _DatasetSingleImage__additional_parameters(self, **hyperparams):
        """ Just sees if patch size is odd or even """
        self.p_odd = self.patch_size % 2

    def _DatasetSingleImage__create_indices(self, mask):
        """ Finds all possible patches """
        x_pos, y_pos = np.nonzero(mask)
        indices = np.array([
            (x, y) for x, y in zip(x_pos, y_pos)
            if self.p <= x <= self.img.shape[0] - self.p and self.p <= y <= self.img.shape[1] - self.p
        ])
        return indices

    def _DatasetSingleImage__get_data(self, idx):
        """ Returns a patch of data based on index """
        x, y = self.indices[idx]
        x1, y1 = x - self.p, y - self.p
        x2, y2 = x + self.p + self.p_odd, y + self.p + self.p_odd

        patch = self.img[x1:x2, y1:y2]
        label = self.lab[x1:x2, y1:y2]
        return patch, label


class DatasetDisjoint(_DatasetSingleImage):
    """
    Creates a Torch Dataset for a single image using disjoint patches

    This type of Dataset is similar to Patches in that it takes in and returns a patch of the image.
    But unlike patches, all the patches are disjoint, so that there is no overlap in training and testing pixels.
    """

    def _DatasetSingleImage__additional_parameters(self, **hyperparams):
        """ Just sees if patch size is odd or even """
        # self.supervision = 'all'
        self.p_odd = self.patch_size % 2

    def _DatasetSingleImage__create_indices(self, mask):
        self.img_og = np.copy(self.img)
        # Calculates how many rows and columns to remove
        rs = self.img.shape[0] % self.patch_size
        cs = self.img.shape[1] % self.patch_size
        # finds the indices to remove from the image to make it divisible by the patch size
        ti, bi = rs // 2 + rs % 2, self.img.shape[0] - rs // 2
        li, ri = cs // 2 + cs % 2, self.img.shape[1] - cs // 2
        self.img = self.img[ti:bi, li:ri, :]
        mask = mask[ti:bi, li:ri]
        ids = [(self.patch_size * x + self.p, self.patch_size * y + self.p)
               for x in range(self.img.shape[0] // self.patch_size)
               for y in range(self.img.shape[1] // self.patch_size)]
        indices = []
        for x, y in ids:
            _, __, msk = self.__get_patch(x, y, mask)
            if np.all(msk == 1):
                indices.append((x, y))
        return indices

    def _DatasetSingleImage__get_data(self, idx):
        """ Returns a patch of data based on index """
        x, y = self.indices[idx]
        return self.__get_patch(x, y)

    def __get_patch(self, x, y, mask=None):
        x1, y1 = x - self.p, y - self.p
        x2, y2 = x + self.p + self.p_odd, y + self.p + self.p_odd

        patch = self.img[x1:x2, y1:y2]
        label = self.lab[x1:x2, y1:y2]
        if mask is not None:
            return patch, label, mask[x1:x2, y1:y2]
        return patch, label
