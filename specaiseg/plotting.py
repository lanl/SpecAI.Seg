import os
from functools import partial
from multiprocessing.pool import ThreadPool

import matplotlib.pyplot as plt
import numpy as np
from scipy import ndimage
from skimage.segmentation import mark_boundaries

from .metrics import distance as dist
from .uq.suqs import suqsES
from .utils import get_pixels




def _mask_image(img, val):
    """Mask the image to only keep val

    Given an image, (specifically one after mark_boundaries),
    then mask all pixels not equal to val.
    This is used to mask all non boarder pixels

    Args:
        img (ndarray): a 3D image matrix, color channel on axis -1
        val (array like): the value to not mask in the image

    Returns:
        masked array: same dimension as image, but masked where not val.
    """
    mask = np.full_like(img, True, dtype=bool)
    new_val = np.full_like(val, False, dtype=bool)
    for x in range(img.shape[0]):
        for y in range(img.shape[1]):
            if np.all(img[x, y, :] == val):
                mask[x, y, :] = new_val
    return np.ma.masked_where(mask, img)


def plot_segmentations(method, datasets, name, num_thread=4, **params):
    """To plot a segmentation method over a set of datasets.

    For a single method (all using the same parameters), it creates the segmentations for each dataset.
    It may be better to use plot_seg instead of this function

    Args:
        method (Segmenter): a class of segmenter
        datasets (dict): where key is name, and value is the dataset
        name (str): name of the segmentation algorithm you are using

    Returns:
        fig: a matplotlib figure

        Typical usage example:
        from HSIsuqs import datasets
        from HSIsuqs.models.segmenters import Slic
        HSIdata = {'IndianPines' : datasets.get_data('IndianPines'),
                'PaviaU' : datasets.get_data('PaviaU'),
                'Salinas' : datasets.get_data('salinas'),
                'KSC' : datasets.get_data('KSC'),
                'PaviaC' : datasets.get_data('paviac'),
                'Botswana' : datasets.get_data('botswana')}
        plot_segmentations(Slic, HSIdata, 'SLIC')
        plt.show()
    """
    fig, axs = plt.subplots(3, 2)
    axs = axs.flatten()

    def wrapper(ds, method, params):
        print(f'Starting: {ds["name"]}')
        seg = method(ds, **params).get_segmentation()
        print(f'Finshed: {ds["name"]}')
        return seg

    with ThreadPool(num_thread) as p:
        kwargs = {'method': method, 'params': params}
        res = p.map(partial(wrapper, **kwargs), datasets.values())
        res = list(res)

    i = 0
    for i in range(len(datasets)):
        # seg = method(ds, **params).segment()
        k = list(datasets.keys())[i]
        ds = datasets[k]
        seg = res[i]
        bounds = mark_boundaries(ds['img_rgb'], seg)
        bounds = (bounds - np.min(bounds)) / (np.max(bounds) - np.min(bounds))
        axs[i].imshow(bounds, aspect="auto")
        axs[i].title.set_text(k)
        i += 1
    fig.set_size_inches(10, 14)
    fig.suptitle(name)
    return fig

def plot_image(ds, folder='./Figures', save=False, fig_size=(5, 5), ds_name=None, rotation=0):
    """Just plots an image

    Essentially wraps plt.imshow, but uses the same parameters as the other
    plotting functions here, like rotation and fig_size.

    Args:
        ds (ndarray or dict): and image, or dict with 'img' key.
        folder (str, optional): folder to save to. Defaults to './Figures'.
        save (bool, optional): whether to save, if string the file name to save. Defaults to False.
        fig_size (tuple, optional): figure size in inches. Defaults to (5, 5).
        ds_name (str, optional): string for the title. Defaults to None.
        rotation (int, optional): degrees to rotate image. Defaults to 0.
    """
    if isinstance(ds, dict):
        img = ds['img']
    else:
        img = ds

    fig = plt.figure()
    fig.set_size_inches(fig_size)
    fig.patch.set_facecolor('white')

    img = ndimage.rotate(img, rotation)
    plt.imshow(img)
    plt.title(f'Image: {ds_name}')

    if save is not None and save is not False:
        if not os.path.isdir(folder):
            os.mkdir(folder)

        if isinstance(save, str):
            name = save
        else:
            name = ds_name

        if not os.path.isdir(folder):
            os.mkdir(folder)
        file = os.path.join(folder, name)
        plt.savefig(file)
        plt.close()


def plot_seg(ds, alg, seg=None, folder='./Figures', save=False,
             alg_args={}, fig_size=(5, 5), rgb_channels=[0, 1, 2],
             bound_args={'mode': 'subpixel'}, ds_name=None, rotation=0):
    """Plot a single segmentation

    ds should be either the dataset or just the image.
    If just passing the image, be sure to set ds_name and rgb_channels appropriatley.
    save can be a file name, or if not specify, then the file name is created
    as datasetname_algname_param_val...param_val_numsegs.png
    The folder it is saved in is folder/datasetname/algname/filename.png.
    To make boarderlines more visible, consider bounds_args={'mode': 'thick'}

    Args:
        ds (dict or ndarray, optional): a dictionary with at least an 'img' key with 3D hyperspectral image, or just the 3D array.
        alg (Seger): the segmentation algorithm used, or to use.
        seg (ndarray, optional): a segmentation if already created. Defaults to None.
        folder (str, optional): the folder where to save image. Defaults to './Figures'.
        save (bool or str, optional): if true then saves, or if string then uses it as file name. Defaults to False.
        alg_args (dict, optional): dict of options to pass to alg, or to create file name. Defaults to {}.
        fig_size (tuple, optional): inches dimension of image. Defaults to (5, 5).
        rgb_channels (list, optional): what indices to use for the false coloring. Defaults to [0, 1, 2].
        bound_args (dict, optional): arguments to pass to mark_boundaries. Defaults to {'mode':'subpixel'}.
        ds_name (str, optional): name of dataset if not included in ds. Defaults to None.
        rotation (float, optional): the degree to rotate the image (90 is a good option for tall/long images). Defaults to 0.
    """

    if isinstance(ds, dict):
        img = ds['img']
        img_rgb = ds['img_rgb']
        ds_name = ds['name']
    else:
        img = ds
        img_rgb = ds[:, :, rgb_channels]
        if ds_name is None:
            raise RuntimeError(f'Need ds_name to not be None')

    if seg is None:
        seg = eval("alg(img, **alg_args).get_segmentation()")

    n_segs = len(np.unique(seg))
    seg_bs = mark_boundaries(img_rgb, seg, **bound_args)
    alg_name = str(alg(None))

    fig = plt.figure()
    fig.set_size_inches(fig_size)
    fig.patch.set_facecolor('white')

    seg_bs = ndimage.rotate(seg_bs, rotation)
    plt.imshow(seg_bs)
    plt.title(f'{ds_name}, alg: {alg_name}, segs: {n_segs}')

    if save is not None and save is not False:
        if not os.path.isdir(folder):
            os.mkdir(folder)

        if isinstance(save, str):
            name = save
        else:
            name = f'{ds_name}_{alg_name}_'
            for k, v in alg_args.items():
                name += f'{str(k)}_{str(v)}_'
            name += f'{n_segs:05d}.png'

        new_path = os.path.join(folder, ds_name)
        if not os.path.isdir(new_path):
            os.mkdir(new_path)
        new_path = os.path.join(new_path, alg_name)
        if not os.path.isdir(new_path):
            os.mkdir(new_path)
        file = os.path.join(new_path, name)
        plt.savefig(file)
        plt.close()


def plot_SUQ(ds, alg, p=.65, r=2, n=50, seg=None, uq=None, dist_func='cos', folder='./Figures/SUQ/',
             alg_args={}, uq_args={}, save=False, title='SUQ', vmin=.2, vmax=1.4, fig_size=(5, 5),
             rotation=0, top_to_bottom=False, colorbar=True, bound_args={'mode': 'subpixel'}):
    """Plot SUQ scores for an image

    If you don't provide a seg, then one will be created using the alg and alg_args.
    save can be a file name, or one will be created if save=True.
    This function can also be used to plot other per region score information, 
    such as fractal dimension, or zeb/frc score. 
    All you need is uq to be a dict with region label, score pairs.
    Output image will be (3*fig_ratio_in) X fig_ratio_in inches.

    Args:
        ds (dict or ndarray, optional): a dictionary with at least an 'img' key with 3D hyperspectral image, or just the 3D array.
        alg (Segmenter): the segmenter used, or to use
        p (float, optional): see suqsES documentation. Defaults to .65.
        r (int, optional): see suqsES documentation. Defaults to 2.
        n (int, optional): see suqsES documentation. Defaults to 50.
        seg (ndarray, optional): if None, a segmentation will be created. Defaults to None.
        uq (dict, optional): uq dict, or if None then one will be created. Defaults to None.
        dist_func (str, optional): see suqsES documentation. Defaults to 'cos'.
        folder (str, optional): Where to save the image. Defaults to './Figures/SUQ/'.
        alg_args (dict, optional): arguments used by the segmenter. Defaults to {}.
        uq_args (dict, optional): additional args sent to suqsES. Defaults to {}.
        save (bool, optional): wether to save, and maybe file name. Defaults to False.
        title (str, optional): Name of Score. Defaults to 'SUQ'.
        vmin (float, optional): minimum value for plotting. Defaults to .2.
        vmax (float, optional): maximum value for plotting. Defaults to 1.4.
        fig_size (tuple, optional): width and height in inches of each subplot. Defualts to (5, 5)
        rotation (float, optional): the degree to rotate the image (90 is a good option for tall/long images). Defaults to 0.
        top_to_bottom (bool, optional): wether to plot top to bottom, or left to right. Defaults to False (left to right).
        colorbar (bool, optional): wether to add colorbar to UQ, may look better without when doing top_to_bottom=True. Defaults to False.
        bound_args (dict, optional): arguments to pass to mark_boundaries. Defaults to {'mode':'subpixel'}.
    """
    if seg is None:
        seg = eval(f'alg(ds, **alg_args).get_segmentation()')
    if uq is None:
        uq_v, uq_d = eval(
            f'suqsES(ds, seg, n=n, p=p, r=r, dist_func=dist.{dist_func}, **uq_args).get_uq()')
    else:
        uq_d = uq
        uq_v = [v for k, v in uq_d.items()]
        if len(uq_v[0]) > 1:
            uq_v = [v[0] for v in uq_v]
    total = seg.size
    uq_w = np.array([np.sum(seg == i) / total for i in uq_d.keys()])

    seg_uq = seg.copy()
    for k in uq_d.keys():
        cond = seg == k
        seg_uq = np.where(cond, uq_d[k][0], seg_uq)
    seg_bs = mark_boundaries(ds['img_rgb'], seg, **bound_args)
    n_segs = len(np.unique(seg))
    ave_score = np.average(uq_v, weights=uq_w)
    mean_score = np.mean(uq_v)

    if top_to_bottom:
        fig, axs = plt.subplots(3, 1)
        fig.set_size_inches((fig_size[0], 3 * fig_size[1]))
    else:
        fig, axs = plt.subplots(1, 3)
        fig.set_size_inches((3 * fig_size[0], fig_size[1]))
    fig.patch.set_facecolor('white')

    seg_bs = ndimage.rotate(seg_bs, rotation)
    axs[0].imshow(seg_bs, aspect="auto")
    axs[0].title.set_text(f'{ds["name"]}, {str(alg(None))}, Segs={n_segs}')

    seg_uq = ndimage.rotate(seg_uq, rotation)
    axs[1].imshow(seg_uq, cmap='hot', aspect='auto', vmin=vmin, vmax=vmax)
    if colorbar:
        pcm = axs[1].pcolormesh(seg_uq, cmap='hot', vmin=vmin, vmax=vmax)
        fig.colorbar(pcm, label=f'{title} Score',
                    orientation='vertical', ax=axs[1])
    axs[1].title.set_text(f'Mean: {mean_score:.4f}, Weighted: {ave_score:.4f}')

    axs[2].hist(uq_v, range=(vmin, vmax))
    axs[2].axvline(mean_score, color='k', linestyle='dashed')
    axs[2].title.set_text(f'Distrubtion of {title} Scores')

    if save is not None and save is not False:
        if isinstance(save, str):
            name = save
        else:
            name = f'{ds["name"]}{str(alg(None))}Segs{n_segs:06d}'

        new_path = os.path.join(folder, ds['name'])
        if not os.path.isdir(new_path):
            os.mkdir(new_path)
        new_path = os.path.join(new_path, dist_func)
        if not os.path.isdir(new_path):
            os.mkdir(new_path)
        new_path = os.path.join(new_path, str(alg(None)))
        if not os.path.isdir(new_path):
            os.mkdir(new_path)
        file = os.path.join(new_path, name)
        plt.savefig(file)
        plt.close()


def plot_seg_spectra(ds, alg, seg=None, folder='./Figures/seg_spectra/', alg_args={}, save=None,
                     cmap_='gist_ncar', img_alpha=0, line_alpha=.7, y_min=0, y_max=1,
                     alg_name=None, rotation=0, top_to_bottom=False, fig_size=(5, 5),
                     bound_args={'mode': 'subpixel'}, rgb_img=None, name=None):
    """Plots each regions average spectra

    If you don't provide a seg, then one will be created using the alg and alg_args.
    save can be a file name, or one will be created if save=True.
    If you set img_alpha greater than 0, it will show the image underneath
    the segmentation coloring, but will distort how those colors look.

    Args:
        ds (dict or ndarray): a dictionary with at least an 'img' key with 3D hyperspectral image, or just the 3D array.
        alg (Segmenter): the segmenter used, or to use
        seg (ndarray, optional): segmentation matrix, or one will be made. Defaults to None.
        folder (str, optional): folder where you want to save the image. Defaults to './Figures/seg_spectra/'.
        alg_args (dict, optional): arguments used to make segmentation. Defaults to {}.
        save (_type_, optional): wether to save and possibly where. Defaults to None.
        cmap_ (str, optional): name of matplotlib cmap. Defaults to 'gist_ncar'.
        img_alpha (int, optional): can show the image under the segment coloring. Defaults to 0.
        line_alpha (float, optional): alpha for the line plot. Defaults to .7.
        y_min (int, optional): min value of spectra. Defaults to 0.
        y_max (int, optional): max value of spectra. Defaults to 1.
        alg_name (str, optional): name of the algorithm. Defaults to None.
        rotation (float, optional): the degree to rotate the image (90 is a good option for tall/long images). Defaults to 0.
        top_to_bottom (bool, optional): wether to plot top to bottom, or left to right. Defaults to False (left to right).
        fig_size (tuple, optional): width and height in inches of each subplot. Defualts to (5, 5).
        bound_args (dict, optional): arguments to pass to mark_boundaries. Defaults to {'mode':'subpixel'}.
        rgb_img (ndarray, optional): if you pass in ds as an image, then you need to pass in this rgb_image. Defaults to None.
        name (str, optional): the name of the dataset, not needed if ds is a dict with 'name' key. Defaults to None.
    """
    if seg is None:
        seg = eval(f'alg(ds, **alg_args).get_segmentation()')

    if isinstance(ds, dict):
        img = ds['img']
        rgb_img = ds['img_rgb']
        name = ds['name']
    else:
        img = ds

    spectras = {}
    max_s = np.max(seg)
    for l in np.unique(seg):
        pixels = get_pixels(l, img, seg)
        spectras[l] = np.mean(pixels, axis=0), l / max_s

    seg_bs = mark_boundaries(rgb_img, seg, **bound_args)
    n_segs = len(np.unique(seg))
    if alg_name is None:
        alg_name = str(alg(None))

    cmap = plt.cm.get_cmap(cmap_)
    if top_to_bottom:
        fig, axs = plt.subplots(3, 1)
        fig.set_size_inches((fig_size[0], 3 * fig_size[1]))
    else:
        fig, axs = plt.subplots(1, 3)
        fig.set_size_inches((3 * fig_size[0], fig_size[1]))
    fig.patch.set_facecolor('white')

    seg_bs = ndimage.rotate(seg_bs, rotation)
    axs[0].imshow(seg_bs, aspect="auto")
    axs[0].title.set_text(f'{name}, {alg_name}, Segs={n_segs}')

    seg = ndimage.rotate(seg, rotation)
    rgb_img = ndimage.rotate(rgb_img, rotation)
    axs[1].imshow(seg, cmap=cmap_, alpha=1, aspect='auto')
    axs[1].imshow(rgb_img, alpha=img_alpha, aspect='auto')
    axs[1].title.set_text(f'Segmentation Coloring')

    axs[2].set_ylim([y_min, y_max])
    for k, v in spectras.items():
        color = cmap(v[1])
        axs[2].plot(v[0], color=color, alpha=line_alpha)
    axs[2].title.set_text(f'Segment Average Spectra')

    if save is not None and save is not False:
        if isinstance(save, str):
            name = save
        else:
            name = f'{name}{str(alg(None))}Segs{n_segs:06d}'

        new_path = os.path.join(folder, name)
        if not os.path.isdir(new_path):
            os.mkdir(new_path)
        new_path = os.path.join(new_path, str(alg(None)))
        if not os.path.isdir(new_path):
            os.mkdir(new_path)
        file = os.path.join(new_path, name)
        plt.savefig(file)
        plt.close()


def plot_individual_SUQ(ds, seg, *reg, uq_args={'n':1, 'p':.65, 'r':2}, seed=0, folder='Figures/SUQ/', save=False):
    """Plots a single simulation of a SUQ score regions

    all parameters after reg need to be named.
    reg can take multiple regions to merge together, though plotting may not look that good.
    save is similar to plot_SUQ save behavior.
    Note that you have to provide a segmentation, one will not be created.

    Args:
        ds (dict or ndarray): a dictionary with at least an 'img' key with 3D hyperspectral image, or just the 3D array.
        seg (ndarray): segmentation matrix
        uq_args (dict, optional): arguments passed to suqES. Defaults to {'n':1, 'p':.65, 'r':2}.
        seed (int, optional): seed for suq simulation. Defaults to 0.
        folder (str, optional): where to save file. Defaults to 'Figures/SUQ/'.
        save (bool or str, optional): whether to save, or the file name. Defaults to False.
    """
    if isinstance(ds, dict):
        img = ds['img_rgb']
    else:
        img = ds
    suq = suqsES(img, seg, **uq_args, seed=seed)
    r = suq.r

    new_reg = reg[0]
    for l in reg:
        seg = np.where(seg == l, new_reg, seg)
    reg = new_reg

    cond = seg == reg
    rows = np.any(cond, axis=1)
    cols = np.any(cond, axis=0)
    rmin, rmax = np.where(rows)[0][[0, -1]]
    cmin, cmax = np.where(cols)[0][[0, -1]]
    rmin, cmin = max(rmin - (r + 3), 0), max(cmin - (r + 3), 0)
    rmax, cmax = min(rmax + (r + 3), seg.shape[0]), min(cmax + (r + 3), seg.shape[1])

    new_img, new_seg, new_cond = img[rmin:rmax, cmin:cmax], seg[rmin:rmax, cmin:cmax], cond[rmin:rmax, cmin:cmax]
    suq = suqsES(new_img, new_seg, **uq_args, seed=seed, disable_min_check=True)

    new_e, new_s, cond, q_e, q_s, q_o, uq = suq.get_e_s_boundaries(reg)
    bounds_og = mark_boundaries(new_img, new_cond, (0, 0, 1), mode='subpixel')
    bounds_e = mark_boundaries(new_img, new_e, (1, 0, 0), mode='subpixel')
    bounds_e = _mask_image(bounds_e, [1, 0, 0])
    bounds_s = mark_boundaries(new_img, new_s, (0, 1, 0), mode='subpixel')
    bounds_s = _mask_image(bounds_s, [0, 1, 0])


    fig, axs = plt.subplots(1, 3)
    fig.set_size_inches(15, 5)
    fig.patch.set_facecolor('white')
    
    axs[0].imshow(mark_boundaries(img, seg, color=(0, 0, 1), mode='subpixel'), aspect='auto')
    axs[0].scatter((cmin+cmax), (rmin+rmax), c='yellow', marker='x')
    axs[0].title.set_text(f'{ds["name"]}, Region: {reg}')

    axs[1].imshow(mark_boundaries(new_img, new_seg, color=(0, 0, 1), mode='subpixel'), aspect='auto')
    axs[1].title.set_text(f'{ds["name"]}, QO: {q_o:.1f}')

    axs[2].imshow(bounds_og, aspect='auto', alpha=1)
    axs[2].imshow(bounds_e[:, :, 0], 'Set1', aspect='auto', alpha=.9)
    axs[2].imshow(bounds_s[:, :, 1], 'Dark2', aspect='auto', alpha=.9)
    axs[2].title.set_text(f'QE: {q_e:.1f}, QS: {q_s:.1f}, SUQ: {uq:.3f}')

    if save is not None and save is not False:
        if isinstance(save, str):
            name = save
        else:
            name = f'{ds["name"]}_{reg}_{seed:06d}'

        new_path = os.path.join(folder, ds['name'])
        if not os.path.isdir(new_path):
            os.mkdir(new_path)
        new_path = os.path.join(new_path, 'SegSUQ')
        if not os.path.isdir(new_path):
            os.mkdir(new_path)
        file = os.path.join(new_path, name)
        plt.savefig(file)
        plt.close()
