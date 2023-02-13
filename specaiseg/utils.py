import json
import os

import numpy as np
import spectral
import torch
from scipy import io, misc, ndimage
from tqdm import tqdm

from ._utils.ReadHSI import Make_RGB_BB, Read_Envi_HSI


class TqdmUpTo(tqdm):
    """ Provides `update_to(n)` which uses `tqdm.update(delta_n)`. """

    def update_to(self, b=1, bsize=1, t_size=None):
        """
        Changes update_to for tqdm

        Parameters
        ----------
            b  : int, optional
                Number of blocks transferred so far [default: 1].
            bsize  : int, optional
                Size of each block (in tqdm units) [default: 1].
            t_size  : int, optional
                Total size (in tqdm units). If [default: None] remains unchanged.
        """
        if t_size is not None:
            self.total = t_size
        self.update(b * bsize - self.n)  # will also set self.n = b * bsize


def get_device(ordinal):
    """Wrapper for torch.device

    Args:
        ordinal (int): number for gpu, or <0 for cpu

    Returns:
        device: return from torch.device()
    """
    # Use GPU ?
    if ordinal < 0:
        print('Computation on CPU')
        device = torch.device('cpu')
    elif torch.cuda.is_available():
        print(f'Computation on CUDA GPU device {ordinal}')
        device = torch.device(f'cuda:{ordinal}')
    else:
        print(
            '/!\\ CUDA was requested but is not available! Computation will go on CPU. /!\\')
        device = torch.device('cpu')
    return device


def open_file(dataset):
    """Opens the file at the given path.

    Supports .mat, .tif/f, and .hdr

    Args:
        dataset (str): path to the file you want to open

    Raises:
        ValueError: if the file extension is not .mat, .tif, or .hdr.

    Returns:
        depends: returns result depending on which file type
    """
    _, ext = os.path.splitext(dataset)
    ext = ext.lower()
    if ext == '.mat':
        # Load Matlab array
        return io.loadmat(dataset)
    elif ext == '.tif' or ext == '.tiff':
        # Load TIFF file
        return misc.imread(dataset)
    elif ext == '.hdr':
        img = spectral.open_image(dataset)
        return img.load()
    elif ext == '':
        img, wav = _read_img_data(dataset)
        return {'img': img, 'wavelength': wav}
    else:
        raise ValueError(f'Unknown file format: {ext}')


def _read_img_data(cube_path, echo=0):
    img_arr = []
    wavelength = []

    if os.path.isfile(cube_path):
        hsi_dir, cube_name = os.path.split(cube_path)
        # HSI Cube Load
        hsi_filename = cube_name + '.hdr'

        img_file = os.path.join(hsi_dir, cube_name)
        hdr_file = img_file + '.hdr'

        if os.path.isfile(hdr_file) and os.path.isfile(img_file):
            # Open ENVI HSI cube
            data = Read_Envi_HSI(cube_name, hsi_dir, echo_command=echo)
            img_arr = data['hsi']
            wavelength = data['wavelength']

    return img_arr, wavelength

def false_grey_img(img_arr, wavelength, RGB_vals=[11.3, 10.2, 8.6]):
    # find indices for 3 band  image
    rgb_index=[]
    RGB_vals = np.array(RGB_vals)
    rband = wavelength.tolist().index(min(wavelength, key=lambda x:abs(x-RGB_vals[0])))
    rgb_index.append(rband)
    gband = wavelength.tolist().index(min(wavelength, key=lambda x:abs(x-RGB_vals[1])))
    rgb_index.append(gband)
    bband = wavelength.tolist().index(min(wavelength, key=lambda x:abs(x-RGB_vals[2])))
    rgb_index.append(bband)
    # Generate false color & grey images from cube
    hsi_grey, hsi_rgb = Make_RGB_BB(rgb_index, img_arr, wavelength)
    return hsi_grey, hsi_rgb



def get_roi(roi_file, img, img_rgb=None, color=(255, 0 ,0)):
    if isinstance(img, dict):
        img_rgb = img['img_rgb']
        img = img['img']
    roi_dict = _read_json_file(roi_file)
    roi_dict = _roi_annotation(img, roi_dict, img_rgb, color)
    return roi_dict
    

# Function to read JSON file with ROI pixels
def _read_json_file(filename):
    roi_dict = {}
    if filename and os.path.isfile(filename):
        with open(filename, 'r') as jfile:
            jdata = jfile.read()
        roi_dict = json.loads(jdata)
    return roi_dict

# Function to get ROI pixels into numpy array indices
def _roi_annotation(img_arr, roi_dict, img_rgb, color):
    img_arr = ndimage.rotate(img_arr, 90)
    img_arr = np.flip(img_arr, axis=1)

    img_rgb = ndimage.rotate(img_rgb, 90)
    img_rgb = np.flip(img_rgb, axis=1)
    # Get array of x, y indices of ROI pixels
    for k in roi_dict.keys():
        rpixels = roi_dict[k]['PIXELS']
        rname = roi_dict[k]['NAME']
        xx = (np.asarray(rpixels) % img_arr.shape[0]).astype('int')
        yy = np.round(np.asarray(rpixels)/img_arr.shape[0]).astype('int')
        xy = list(zip(xx, yy))
        # roi_dict[k]['XY'] = xy
        pixels = []
        rgb_mask = img_rgb.copy()
        img_mask = np.full((img_arr.shape[0], img_arr.shape[1]), False)
        for i in xy:
            img_mask[i[0], i[1]] = True
            pix = img_arr[i[0], i[1], :]
            pixels.append(pix)
            rgb_mask[i[0], i[1], :] = color
        roi_dict[k]['PIXEL_VALS'] = np.asarray(pixels)
        img_mask = np.flip(img_mask, axis=1)
        img_mask = ndimage.rotate(img_mask, -90)
        rgb_mask = np.flip(rgb_mask, axis=1)
        rgb_mask = ndimage.rotate(rgb_mask, -90)

        roi_dict[k]['RGB_MASK'] = rgb_mask
        roi_dict[k]['IMG_MASK'] = img_mask

    return roi_dict


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
