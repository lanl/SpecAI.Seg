"""
Functions to read HSI data files, with default arguments oriented toward Mako/Urban Vigil.
For instance, this can be useful for reading L2 (processed) or L1 (raw) HSI cubes.

Main function is Read_Envi_HSI, with the remaining functions called by Read_Envi_HSI.

Extracted from pyHAT (see README.md) with modifications/documentation by Natalie Klein.
"""

import re
import os
import math
import matplotlib.pyplot as plt
import numpy as np
import pyparsing as pp
from skimage import exposure
from scipy import ndimage

def Read_Envi_HSI(file, data_path, results_path='', lam_min_set=7.2, lam_max_set=13.8, 
                  spat_min_set=0, spat_max_set=0, frame_min_set=0, frame_max_set=0, 
                  band_removal=1, all_band_removal=0, hsi_file_suffix='L2S.dat', 
                  bad_bands=[], mean_or_med=0, bad_samples=[], echo_command=1):
    """
    From Read_Envi_HSI.py.
    
    Function to read in HSI data files.
    
    Defaults are configured for Mako data; see Settings_Library files for hints on setting
    inputs for other data sources.

    Args:
        file (string): HSI file name 
        data_path (string): HSI file path
        results_path (string): Specify path for saving results. (NEEDED?) Defaults to ''.
        lam_min_set (float): Wavelength range minimum. Defaults to 7.2.
        lam_max_set (float): Wavelength range maximum. Defaults to 13.8.
        spat_min_set (int): Clip spatial pixels; starting point for included pixels. Defaults to 0 (no clip).
        spat_max_set (int): Clip spatial pixels; ending point for included pixels. Defaults to 0 (no clip).
        frame_min_set (int): Clip frames; starting point for included frames. Defaults to 0 (no clip).
        frame_max_set (int): Clip frames; ending point for included frames. Defaults to 0 (no clip).
        band_removal (bool): For spectral fits, remove atmospheric bands (value 1), or don't (value 0). Defaults to 1. (NEEDED?)
        all_band_removal (bool): For all fits, remove atmospheric bands (value 1), or don't (value 0). Defaults to 0. (NEEDED?)
        hsi_file_suffix (string): Suffix for HSI file; defaults to 'L2S.dat'.
        bad_bands (list): List of bad spectral band indices (sensor-specific). Defaults to [] (no bad bands).
        mean_or_med (bool): Use mean (value 0) or median (value 1) for outliers. (NEEDED?)
        bad_samples (list): ist of bad frames. Defaults to [] (no bad frames).
        echo_command (int): Print results to console (value 1) or don't (value 0); value 2 also plots figures. Defaults to 1.

    Returns:
        dict: dictionary of HSI data and information.
    """
    if data_path is None:
        data_path,file = os.path.split(file)

    # define current file and define header
    if 'hsic' in file:
        cube_name = file[0:int(len(file)-5)]
    elif 'sc' in file:
        cube_name = file[0:int(len(file)-3)]
    else:
        cube_name = file.split(".")[0]

    filename = os.path.join(data_path,file)
    
    # check for header file first
    header_check = 1
    headerName = file+'.hdr'
    headerPath = filename+'.hdr'
    if not os.path.exists(headerPath):                                        
        headerName = file.rpartition('.')[0]+'.hdr'     
    file_list = os.listdir(data_path)
    if headerName in file_list:
        try:
            header = Read_Envi_Header(filename)
        except:
            header_check = 0
    else:
        header_check = 0
    
    # bad data flag
    bad_data_flag = 1
    
    # check for info in header
    if header_check == 1:
        interleave_check = 0
        if hasattr(header, 'interleave'):
            interleave_check = 1
        samples_check = 0
        if hasattr(header, 'samples'):
            samples_check = 1
        lines_check = 0
        if hasattr(header, 'lines'):
            lines_check = 1
        bands_check = 0
        if hasattr(header, 'bands'):
            bands_check = 1
        bad_data_flag*=(interleave_check*samples_check*lines_check*
                        bands_check*header_check)
    else:
        bad_data_flag = 0
                            
    if bad_data_flag == 1:

        # read in image data, consider interleave and byte order
        if header.interleave == 'npy':
            hsi = np.load(filename)
        else:
            if (header.interleave == 'bip' or header.interleave == 'BIP'): odr='C'
            if (header.interleave == 'bil' or header.interleave == 'BIL'): odr='C'
            if (header.interleave == 'bsq' or header.interleave == 'BSQ'): odr='C'
            hsi = np.fromfile(filename, dtype=header.type, sep="")
            if hasattr(header, 'byteorder'):
                if header.byteorder == 1:
                    hsi = hsi.byteswap()
                                        
        # change data to 32 bit float
        hsi = np.float32(hsi)
        if hsi_file_suffix == ".sc":
            hsi = np.nan_to_num(hsi)
                    
        # bad data flag
        if np.any(np.isinf(hsi)):
            bad_data_flag = 0
        elif np.any(np.isnan(hsi)):
            bad_data_flag = 0
        else:
            bad_data_flag = 1
                
        # delete header offset if needed
        if hasattr(header, 'header_offset'):
            if header.header_offset != 0:
                if header.type == 'float64':
                    hsi = hsi[int(header.header_offset/8):int(hsi.shape[0])]
                elif header.type == 'float32':
                    hsi = hsi[int(header.header_offset/4):int(hsi.shape[0])]
                
        #  reshape based on values from header
        if header.interleave != 'npy':
            if (header.interleave == 'bip' or header.interleave == 'BIP'):
                try:
                    hsi = hsi.reshape([header.lines,header.samples,header.bands], order=odr)
                except:
                    hsi = []
                    bad_data_flag = 0
            if (header.interleave == 'bil' or header.interleave == 'BIL'):
                hsi = hsi.reshape([header.lines,header.bands,header.samples], order=odr)
                hsi = np.swapaxes(hsi,1,2)
            if (header.interleave == 'bsq' or header.interleave == 'BSQ'):
                hsi = hsi.reshape([header.bands,header.lines,header.samples], order=odr)
                hsi = np.swapaxes(hsi,0,2)
                hsi = np.swapaxes(hsi,0,1)
               
         # description
        if hasattr(header, 'description'):
            description = header.description
        else:
            description = []
            
        # aerospace calibrated L2S.dat rotate by 90
        if len(description) == 0:
            if hsi_file_suffix ==  "L2S.dat" and lam_min_set > 7:
                hsi = hsi[::-1,:,:]
                hsi = hsi[:,::-1,:]
            elif hsi_file_suffix == "L2S.dat" and lam_min_set < 7:
                hsi = hsi[::-1,:,:]
        elif len(description) >= 1:
            try:
                if 'pyHAT' not in description[1]:
                    if hsi_file_suffix == "L2S.dat" and lam_min_set > 7:
                        hsi = hsi[::-1,:,:]
                        hsi = hsi[:,::-1,:]
                    elif hsi_file_suffix == "L2S.dat" and lam_min_set < 7:
                        hsi = hsi[::-1,:,:]
            except:
                if 'pyHAT' not in description[0]:
                    if hsi_file_suffix == "L2S.dat" and lam_min_set > 7:
                        hsi = hsi[::-1,:,:]
                        hsi = hsi[:,::-1,:]
                    elif hsi_file_suffix == "L2S.dat" and lam_min_set < 7:
                        hsi = hsi[::-1,:,:]
                
        # Aces-Hy data
        if hsi_file_suffix == "corrected.hsi":
            hsi = hsi[::-1,:,:]
               
        # Telops data
        if hsi_file_suffix == ".sc":
            wavenumber = np.array(header.wavelength)
            header.wavelength = 1e4/wavenumber[::-1]
            hsi = 1e-2*hsi[:,::,::-1]*wavenumber[np.newaxis,np.newaxis,::-1]**2
            if file[0] != "2" and lam_min_set>7:
                hsi = hsi[:,::-1,:]
                if lam_min_set != 8:
                    lam_min_set = 8
                if lam_max_set != 12.4:
                    lam_max_set = 12.4
            elif file[0] == "2" and lam_min_set>7:
                hsi = hsi[:,::-1,:]  
                if lam_min_set != 8:
                    lam_min_set = 8
                if lam_max_set != 12.4:
                    lam_max_set = 12.4
            if lam_min_set<6:
                hsi = hsi[::-1,:,:] 

        # Headwall data
        if hsi_file_suffix == "":
            # SWIR
            if header.bands<400 and lam_min_set < 2.19:
                hsi = np.swapaxes(hsi,0,1)
                hsi = hsi[::-1,:,:]
                # SWIR
                # if lam_min_set>0.9:
                #    hsi = hsi[:,::-1,:]
            else:
                # hi-res SWIR
                if lam_min_set>2.19:
                    hsi = 1e4*np.swapaxes(hsi,0,1)
                    hsi = hsi[:,::-1,:]
                    hsi = hsi[::-1,:,:]
                    if lam_min_set != 2.2:
                        lam_min_set = 2.2
                    if lam_max_set != 2.49:
                        lam_max_set = 2.49
                # UV
                else:
                    hsi = 1e3*np.swapaxes(hsi,0,1)
                    hsi = hsi[::-1,:,:]
                    if lam_min_set != 0.35:
                        lam_min_set = 0.35
                    if lam_max_set != 0.495:
                        lam_max_set = 0.495
            header.lines = int(hsi.shape[0])
            header.samples = int(hsi.shape[1])
            
        wavelength_loc = np.array(header.wavelength)
        if wavelength_loc[0]>wavelength_loc[wavelength_loc.shape[0]-1]:
            wavelength_loc = np.array(header.wavelength)
            wavelength_loc = wavelength_loc[::-1]
            header.wavelength = wavelength_loc
            hsi = hsi[:,:,::-1]
            
    if bad_data_flag == 1:
            
        data = Initialize_HSI(file, hsi, header, data_path, 
                              results_path, spat_min_set, spat_max_set, 
                              lam_min_set, lam_max_set, hsi_file_suffix, 
                              bad_bands, bad_samples, frame_min_set, frame_max_set,
                              mean_or_med, band_removal, all_band_removal,
                              bad_data_flag, echo_command, gui=0)     
        atm_line_index = data['atm_line_index']
        specIndex = data['specIndex']
        rgb_index = data['rgb_index']
        hsi = data['hsi']
        wavelength = data['wavelength']
        fwhm = data['fwhm']
        bad_data_flag = data['bad_data_flag']
        s_atm_line = data['s_atm_line']
        e_atm_line = data['e_atm_line']
        spec_range = data['spec_range']
        RGB_vals = data['RGB_vals']
        dead_bands = data['dead_bands']
        waveMap = data['waveMap']
        rad_mean = data['rad_mean']
        dead_cube_bands = data['dead_cube_bands']
        waveCoef = data['waveCoef']
        dead_cube_samples = data['dead_cube_samples']
        dead_samples = data['dead_samples']
        center_wavelength = data['center_wavelength']
        broad = data['broad']
        
    else:
        
        cube_name = []
        hsi = []
        header = []
        rgb_index = []
        wavelength = []
        fwhm = []
        rad_mean = []
        atm_line_index = []
        specIndex = []
        s_atm_line = []
        e_atm_line = []
        waveMap = []
        dead_cube_samples = []
        dead_cube_bands = []
        dead_samples = []
        dead_bands = []
        center_wavelength = []
        description = []
        waveCoef = []
        spec_range = []
        RGB_vals = []
        broad = []
                                
    return {'cube_name':cube_name, 'hsi':hsi, 'header':header, 'rgb_index':rgb_index, 
            'wavelength':wavelength, 'fwhm': fwhm, 'rad_mean': rad_mean, 
            'atm_line_index': atm_line_index, 'specIndex':specIndex, 'dead_cube_bands':dead_cube_bands,
            'bad_data_flag':bad_data_flag, 's_atm_line':s_atm_line, 'e_atm_line':e_atm_line, 
            'description':description, 'waveMap':waveMap, 'waveCoef':waveCoef,
            'spec_range':spec_range,'dead_cube_samples':dead_cube_samples,'RGB_vals':RGB_vals, 
            'dead_samples':dead_samples,'dead_bands':dead_bands, 'center_wavelength':center_wavelength,
            'broad':broad}
    
def Make_RGB_BB(rgb_index, hsi, wavelength):
    """ From Make_RGB_BB.py. (Modified/simplified)
    
    Makes false RGB image from HSI data. 

    Args:
        rgb_index (list): indices to use for false RGB
        hsi (ndarray): HSI data cube
        wavelength (array): HSI data wavelengths

    Returns:
        ndarrays: greyscale and RGB image arrays
    """
    
    lam_min = wavelength[0]
    lam_max = wavelength[len(wavelength)-1]
    
    #contruct broadband
    broadband_kern = np.sum(hsi, axis=2)/hsi.shape[2]
    broadband_eq = exposure.equalize_hist(broadband_kern)
    broadband = exposure.adjust_gamma(broadband_eq, gamma=1, gain=1)  
    
    # broadband_loc = ndimage.rotate(broadband, 90)
    # broadband_loc = np.flip(broadband_loc,axis=1)
    
    #histogram clip and gamma stretch    
    if (lam_min>=0.3 and lam_max<=0.6):
        gamma_set = 0.50
    # VNIR
    elif (lam_min>=0.35 and lam_max<=1.2):
        gamma_set = 0.75
    #VNIRSWIR
    elif (lam_min>=0.35 and lam_min<1.7 and lam_max>=2.3):
        gamma_set = 0.75
    #SWIR
    elif (lam_min<1.6 and lam_min>0.9 and lam_max<=2.55 and lam_max>1.75): 
        gamma_set = 0.5
    #SWIR 2+
    elif (lam_min<2.2 and lam_min>1.9 and lam_max<=2.55 and lam_max>2.3): 
        gamma_set = 0.5
    #SWIR 2
    elif (lam_min>2.2 and lam_max<=2.55 and lam_max>2.3): 
        gamma_set = 0.5
    #LWIR
    elif (lam_min<=9.5 and lam_min>6.9 and lam_max<=14.5 and lam_max>10):
        gamma_set = 0.75
    #MWIR
    elif (lam_min<=4.6 and lam_min>3 and lam_max<=5.6 and lam_max>3.6):
        gamma_set = 0.75
        
    #averaging weight
    rgb_width = 10
    sigma = rgb_width/(2*math.sqrt(2*math.log(2)))
    gauss_x = np.arange(0,wavelength.shape[0],1)
    
    #red channel
    gauss_y = np.exp(-(gauss_x-rgb_index[0])**2/(2*sigma**2))
    rslice = np.mean(abs(hsi*gauss_y[np.newaxis,np.newaxis,:]),axis=2)
    min_val = np.amin(np.amin(rslice,axis=1),axis=0)
    max_val = np.amax(np.amax(rslice,axis=1),axis=0)
    if (max_val == min_val):
        min_val = 0
        max_val = 1
    rslice = (rslice - min_val)/(max_val-min_val)
    rslice_eg = exposure.equalize_hist(rslice)
    rslice_eq = exposure.adjust_gamma(rslice_eg, gamma=gamma_set, gain=1)
        
    #green channel
    gauss_y = np.exp(-(gauss_x-rgb_index[1])**2/(2*sigma**2))
    gslice = np.mean(abs(hsi*gauss_y[np.newaxis,np.newaxis,:]),axis=2)
    min_val = np.amin(np.amin(gslice,axis=1),axis=0)
    max_val = np.amax(np.amax(gslice,axis=1),axis=0)
    if (max_val == min_val):
        min_val = 0
        max_val = 1
    gslice = (gslice - min_val)/(max_val-min_val)
    gslice_eq = exposure.equalize_hist(gslice)
    gslice_eq = exposure.adjust_gamma(gslice_eq, gamma=gamma_set, gain=1)

    #blue channel
    gauss_y = np.exp(-(gauss_x-rgb_index[2])**2/(2*sigma**2))
    bslice = np.mean(abs(hsi*gauss_y[np.newaxis,np.newaxis,:]),axis=2)
    min_val = np.amin(np.amin(bslice,axis=1),axis=0)
    max_val = np.amax(np.amax(bslice,axis=1),axis=0)
    if (max_val == min_val):
        min_val = 0
        max_val = 1
    bslice = (bslice - min_val)/(max_val-min_val)
    bslice_eq = exposure.equalize_hist(bslice)
    bslice_eq = exposure.adjust_gamma(bslice_eq, gamma=gamma_set, gain=1)
    
    # construct false color image
    rgb = np.zeros((hsi.shape[0], hsi.shape[1], 3),'uint8')
    rgb[..., 0]=255*rslice_eq
    rgb[..., 1]=255*gslice_eq
    rgb[..., 2]=255*bslice_eq
    
    # rgb_loc = ndimage.rotate(rgb, 90)
    # rgb_loc = np.flip(rgb_loc,axis=1)

    # return broadband_loc, rgb_loc
    return broadband, rgb

def Initialize_HSI(file, hsi, header, data_path, 
                   results_path, spat_min_set, spat_max_set, 
                   lam_min_set, lam_max_set, hsi_file_suffix, 
                   bad_bands, bad_samples, frame_min_set, frame_max_set,
                   mean_or_med, band_removal, all_band_removal,
                   bad_data_flag, echo_command, gui):
    """ From Initialize_HSI.py.

    Args:
        file ([type]): [description]
        hsi ([type]): [description]
        header ([type]): [description]
        data_path ([type]): [description]
        results_path ([type]): [description]
        spat_min_set ([type]): [description]
        spat_max_set ([type]): [description]
        lam_min_set ([type]): [description]
        lam_max_set ([type]): [description]
        hsi_file_suffix ([type]): [description]
        bad_bands ([type]): [description]
        bad_samples ([type]): [description]
        frame_min_set ([type]): [description]
        frame_max_set ([type]): [description]
        mean_or_med ([type]): [description]
        band_removal ([type]): [description]
        all_band_removal ([type]): [description]
        bad_data_flag ([type]): [description]
        echo_command ([type]): [description]
        gui ([type]): [description]
    """
    
     #cube dimensions
    if echo_command >= 1:
        print("Lines:", hsi.shape[0])
        print("Samples:", hsi.shape[1])
        print("Bands:", hsi.shape[2])
                                 
    #check for pyHAT generated wave_coef file
    check_wavecoef = 0
    list_of_files = []
    if os.path.exists(results_path):
        for x in os.listdir(results_path):
            if "wavecoef_pyHAT" in x:
                temp = results_path + os.sep + x
                list_of_files.append(temp)
                check_wavecoef = 0
        if check_wavecoef == 1:
            if echo_command >= 1:
                print("Using pyHAT wavelength coeficients")
            temp = max(list_of_files, key=os.path.getctime)   
            wavecoef_file = temp
            f = open(wavecoef_file, 'r')
            wavecoef = f.readlines()
            wavecoef = np.array(wavecoef).tolist()
            if len(wavecoef)<8:
                for i in range(int(8-len(wavecoef))):
                    wavecoef.append(0.0)
            wave_coef = [wavecoef[0],wavecoef[1],wavecoef[2],wavecoef[3],
                          wavecoef[4],wavecoef[5],wavecoef[6],wavecoef[7]]
            header.wavecoef = np.array(wave_coef).astype(np.float)
            
    # check for aux file wavemap, fwhm, and broadening
    check_aux_wavemap = 0
    if '.hsic' in file:
        aux_name = file.replace('hsic','aux')
    elif 'L1.dat' in file:
        aux_name = file.replace('L1','aux')
    elif 'L2S.dat' in file:
        aux_name = file.replace('L2S','aux')
    else:
        aux_name = 'dummy'
    center_wavelength = []
    if os.path.exists(os.path.join(data_path,aux_name)):
        aux_data = Read_Envi_Aux(os.path.join(data_path,aux_name), echo_command)
        waveMap = aux_data['wavelength_map']
        if len(waveMap)>0:
            check_aux_wavemap = 1
            if echo_command >= 1:
                print("Using aux wavemap")
    if check_aux_wavemap == 1:
        if hasattr(header, 'fwhm') == True:
            fwhm = header.fwhm
        elif hasattr(header, 'sigma') == True:
            fwhm = header.sigma*(2*np.sqrt(2*math.log(2)))
        if hasattr(header, 'broadcoef') == True:   
            broad = list(header.broadcoef)
        if hasattr(header, 'wavecoef') == True:   
            waveCoef = list(header.wavecoef)
            check_wavecoef = 1
        
    # check for removed bands and samples
    dead_bands = np.ones(hsi.shape[2], dtype=bool)
    dead_samples = np.ones(hsi.shape[1], dtype=bool)
    dead_cube_bands = []
    dead_cube_samples = []
    if check_aux_wavemap == 0:
        if hasattr(header, 'removed_samples'):
            removed_samples = list(header.removed_samples)
            dead_samples = np.ones(int(header.samples+len(removed_samples)), dtype=bool)
            dead_samples[removed_samples] = False 
            dead_cube_samples = np.where(dead_samples == False)            
        if hasattr(header, 'removed_bands'):
            removed_bands = list(header.removed_bands)
            dead_bands = np.ones(int(header.bands+len(removed_bands)), dtype=bool)
            dead_bands[removed_bands] = False
            dead_cube_bands = np.where(dead_bands == False) 
    
    # make wavemap from coeficients
    if hasattr(header, 'wavecoef') and check_aux_wavemap == 0:
        
        if echo_command >= 1 and check_aux_wavemap == 0:
            print("Generating wavelength map from coeficients")
        elif echo_command >= 1 and check_aux_wavemap == 1:
            print("Loaded wavelength map from aux file")

        # get wavecoef out of header
        wavecoef = header.wavecoef
        waveCoef = wavecoef.tolist()
        check_wavecoef = 1
                        
        if len(waveCoef)<8:
            for i in range(int(8-len(waveCoef))):
                waveCoef.append(0.0)
                
        # check order because Aerospace can't be consistent
        if hasattr(header, 'sensor_type') and 'L2' in hsi_file_suffix:
            # Mako
            if "_C" not in header.sensor_type and 'MAKO' in header.sensor_type:
                waveCoef_temp = np.zeros(len(waveCoef))
                waveCoef_temp[0] = waveCoef[0]
                waveCoef_temp[1] = waveCoef[1]
                waveCoef_temp[2] = waveCoef[3]
                waveCoef_temp[3] = waveCoef[4]
                waveCoef_temp[4] = waveCoef[2]
                waveCoef_temp[5] = waveCoef[6]
                waveCoef_temp[6] = waveCoef[5]
                waveCoef_temp[7] = 0.0
                waveCoef = waveCoef_temp
            # Mahi
            if "_C" not in header.sensor_type and 'MAHI' in header.sensor_type:
                waveCoef_temp = np.zeros(len(waveCoef))
                waveCoef_temp[0] = waveCoef[0]
                waveCoef_temp[1] = waveCoef[1]
                waveCoef_temp[2] = waveCoef[2]
                waveCoef_temp[3] = waveCoef[3]
                waveCoef_temp[4] = waveCoef[4]
                waveCoef_temp[5] = waveCoef[5]
                waveCoef_temp[6] = waveCoef[6]
                waveCoef_temp[7] = 0.0
                waveCoef = waveCoef_temp
            # Rainbow Trout
            if ("_C" not in header.sensor_type and 'MNTR' in header.sensor_type 
                or 'SEBASS' in  header.sensor_type or 
                'SHARP' in  header.sensor_type):
                waveCoef_temp = np.zeros(len(waveCoef))
                waveCoef_temp[0] = waveCoef[0]
                waveCoef_temp[1] = waveCoef[1]
                waveCoef_temp[2] = waveCoef[3]
                waveCoef_temp[3] = waveCoef[4]
                waveCoef_temp[4] = waveCoef[2]
                waveCoef_temp[5] = waveCoef[6]
                waveCoef_temp[6] = waveCoef[5]
                waveCoef_temp[7] = 0.0
                waveCoef = waveCoef_temp
                
        # fill wavemap
        if check_aux_wavemap == 0:
            spec_vec = np.float64(np.arange(0,dead_bands.shape[0],1))
            spat_vec = np.float64(np.arange(0,dead_samples.shape[0],1))
            waveMap = np.zeros((spat_vec.shape[0], spec_vec.shape[0]))
            waveMap = (waveCoef[0] + 
                        waveCoef[1]*spat_vec[:,np.newaxis] +
                        waveCoef[2]*spec_vec[np.newaxis,:] +
                        waveCoef[3]*spec_vec[np.newaxis,:]*spat_vec[:,np.newaxis] +
                        waveCoef[4]*spat_vec[:,np.newaxis]**2 + 
                        waveCoef[5]*spec_vec[np.newaxis,:]**2 +
                        waveCoef[6]*spec_vec[np.newaxis,:]*spat_vec[:,np.newaxis]**2+
                        waveCoef[7]*spec_vec[np.newaxis,:]**2*spat_vec[:,np.newaxis])
        
        # square root bullshit
        if hasattr(header, 'sensor_type'):
            if ('MNTR' in header.sensor_type or 'SEBASS' in header.sensor_type
                or 'SHARP' in header.sensor_type):
                  waveMap = np.sqrt(waveMap)
        if waveMap[0,0]>10:
            waveMap = np.sqrt(waveMap)
                                 
    # generate uniform map
    if check_wavecoef == 0 and check_aux_wavemap == 0:
        # convert wavelength and fwhm from nanometers to microns
        lam_min = header.wavelength[0]
        if (lam_min > 10):
            header.wavelength /= 1000.0
        if echo_command >= 1:
            print("Generating uniform wavelength map")
        waveCoef = []
        waveMap = np.tile(header.wavelength, (header.samples, 1))
            
    # center wavelengths
    if len(waveCoef)>8:
        waveCoef = waveCoef[0:8]
    center_wavelength = waveMap[int(waveMap.shape[0]/2),:]
            
    # get band spacing
    if check_aux_wavemap == 0:
        del_lam =  np.zeros(len(center_wavelength),dtype='float')
        for i in range(len(center_wavelength)): 
            if (i == 0):
                del_lam[i] = center_wavelength[i+1]-center_wavelength[i]
            else:
                del_lam[i] = center_wavelength[i]-center_wavelength[i-1]
        
    # make broadening vector
    # alpha = del_lam^2/(2*sigma**2)
    # alpha = 4*math.log(2)*del_lam^2/fwhm**2
    # alpha = 4*math.log(2)/broad_vec**2
    # broad_vec = np.sqrt(4*math.log(2)/alpha)
    # for files with broadening coeficient
    if check_aux_wavemap == 0:
        broad_fit = 0
        if hasattr(header, 'broadcoef') == True and hasattr(header, 'fwhm') == False and hasattr(header, 'sigma') == False:
            broad = header.broadcoef
            spec_vec = np.float64(np.arange(0,len(center_wavelength),1))
            broad_vec = broad[0] + broad[1]*spec_vec + broad[2]*spec_vec**2 + broad[3]*spec_vec**3 + broad[4]*spec_vec**4
            fwhm = broad_vec*del_lam
        elif hasattr(header, 'broadcoef') == False and hasattr(header, 'fwhm') == False and hasattr(header, 'sigma') == False:
            # Mako
            if center_wavelength[0]>7 and hsi_file_suffix == "L2S.dat":
                broad = [2*np.sqrt(math.log(2)/1.22),0,0,0,0]
            # Mahi
            elif (center_wavelength[0]>3 and center_wavelength[0]<5 and hsi_file_suffix == "L2S.dat"):
                broad = [1.1,0,0,0,0]
            else:
                broad = [1.0,0,0,0,0]
            spec_vec = np.float64(np.arange(0,len(center_wavelength),1))
            broad_vec = broad[0] + broad[1]*spec_vec + broad[2]*spec_vec**2 + broad[3]*spec_vec**3 + broad[4]*spec_vec**4
            fwhm = broad_vec*del_lam
        elif hasattr(header, 'broadcoef') == True and hasattr(header, 'fwhm') == True and hasattr(header, 'sigma') == False:
            broad = header.broadcoef
            fwhm = header.fwhm
        elif hasattr(header, 'broadcoef') == True and hasattr(header, 'fwhm') == False and hasattr(header, 'sigma') == True:
            broad = header.broadcoef
            sigma = header.sigma
            fwhm = sigma*(2*np.sqrt(2*math.log(2)))
        elif hasattr(header, 'broadcoef') == False and hasattr(header, 'fwhm') == True and hasattr(header, 'sigma') == False:
            fwhm = header.fwhm
            if fwhm[0]>1:
                fwhm/=1000
            # Pahsi/BG
            if center_wavelength[0]>7 and hsi_file_suffix == ".hsic":
                broad_ratio = fwhm/(del_lam*np.sqrt(0.8))
                broad_fit = 1
            # Mako
            elif center_wavelength[0]>7 and hsi_file_suffix == "L2S.dat":
                broad = [2*np.sqrt(math.log(2)/1.22),0,0,0,0]
            # Mahi
            elif (center_wavelength[0]>3 and center_wavelength[0]<5 and hsi_file_suffix == "L2S.dat"):
                broad = [1.1,0,0,0,0]
            else:
                broad_ratio = fwhm/del_lam
                broad_fit = 1
        elif hasattr(header, 'broadcoef') == False and hasattr(header, 'sigma') == True and hasattr(header, 'fwhm') == False:
            sigma = header.sigma
            fwhm = sigma*(2*np.sqrt(2*math.log(2)))
            # Pahsi
            if center_wavelength[0]>7 and hsi_file_suffix == ".hsic":
                broad_ratio = fwhm/(del_lam*np.sqrt(0.8))
                broad_fit = 1
                                    
        # fit broad coeficients
        if broad_fit == 1:
            broad_ratio = fwhm/del_lam
            broad_ratio_x = np.float64(np.arange(0,broad_ratio.shape[0],1))
            fit = np.polyfit(broad_ratio_x,broad_ratio,4)
            broad = [fit[4],fit[3],fit[2],fit[1],fit[0]]
                    
    # flag pixels outside of specified wavelength range
    a = np.argmin(np.abs(center_wavelength - lam_min_set))
    b = np.argmin(np.abs(center_wavelength - lam_max_set))
    if b == center_wavelength.shape[0]-1:
        b+=1
    dead_bands[0:a] = False
    dead_bands[b:len(dead_bands)] = False
    
      # add pre-determined bad bands to dead_bands list
    dead_bands[bad_bands] = False
    dead_samples[bad_samples] = False

    # if neeed remove spatial pixels or frames
    if spat_min_set != 0 or spat_max_set != 0:
        dead_samples[0:spat_min_set] = False
        dead_samples[spat_max_set:len(dead_samples)] = False
       
    # crop wavelengths in range
    wavelength = center_wavelength[dead_bands]
    waveMap = waveMap[dead_samples,:]
    waveMap = waveMap[:,dead_bands]
    
    # make new fwhm or check for in header
    if check_aux_wavemap == 0:
        del_lam_loc = del_lam[dead_bands]
        spec_vec = np.float64(np.arange(0,len(center_wavelength),1))
        spec_vec_loc = spec_vec[dead_bands]
        broad_vec = broad[0] + broad[1]*spec_vec_loc + broad[2]*spec_vec_loc**2 + broad[3]*spec_vec_loc**3 + broad[4]*spec_vec_loc**4
        fwhm = broad_vec*del_lam_loc
                        
    # crop hsi in range
    bad_bands_loc = np.where(dead_bands == False)
    if len(bad_bands_loc[0])>0 and bad_data_flag == 1:
        dead_bands = np.delete(dead_bands,dead_cube_bands)
        hsi = hsi[:,:,dead_bands]
    
    bad_samples_loc = np.where(dead_samples == False)
    if len(bad_samples_loc[0])>0 and bad_data_flag == 1:
        dead_samples = np.delete(dead_samples,dead_cube_samples)
        hsi = hsi[:,dead_samples,:]
    
    if bad_data_flag == 1:
        header.samples = int(hsi.shape[1])
        header.bands = int(hsi.shape[2])
    if echo_command >= 1 and bad_data_flag == 1:
        print("Samples in specified range:", hsi.shape[1])
        print("Bands in specified range:", hsi.shape[2])
            
    if frame_min_set != 0 or frame_max_set != 0 and bad_data_flag == 1:
        hsi = hsi[frame_min_set:frame_max_set,:,:]
        header.lines = int(hsi.shape[0])
        header.samples = int(hsi.shape[1])
        if echo_command >= 1:
            print("Lines in specified range:", hsi.shape[0])
            
    if hsi_file_suffix == ".sc": 
        hsi[hsi<=0] = 0 
        rad_mean = np.mean(np.mean(hsi,axis=1),axis=0)
        min_mean_thresh = 0.3*np.amin(rad_mean)
        where_neg = np.where(hsi<=min_mean_thresh)
        x_pos = where_neg[0]
        y_pos = where_neg[1]
        z_pos = where_neg[2]
        perc_neg = 100*len(z_pos)/(hsi.shape[0]*hsi.shape[1]*hsi.shape[2])
        if echo_command >= 1:
            print(round(perc_neg,2),"% bad pixels values replaced with mean")
        for i in range(len(x_pos)):
            hsi[x_pos[i],y_pos[i],z_pos[i]] = rad_mean[z_pos[i]]
        
    # calculate mean radiance and assign some stuff
    header.bands = wavelength.shape[0]
    wavelength = wavelength
    if bad_data_flag == 1:
        if mean_or_med == 0:
            rad_mean = np.mean(np.mean(hsi,axis=1),axis=0)
        else:
            rad_mean = np.percentile(np.percentile(hsi,50,axis=1),50,axis=0)
    else:
        rad_mean = []
    
    # get band indices
    results = Get_Band_Indices(hsi, waveMap, wavelength, fwhm ,band_removal, 
                                hsi_file_suffix, all_band_removal, bad_data_flag,
                                dead_bands, echo_command)
    atm_line_index = results['atm_line_index']
    specIndex = results['specIndex']
    rgb_index = results['rgb_index']
    hsi = results['hsi']
    wavelength = results['wavelength']
    fwhm = results['fwhm']
    bad_data_flag = results['bad_data_flag']
    s_atm_line = results['s_atm_line']
    e_atm_line = results['e_atm_line']
    spec_range = results['spec_range']
    RGB_vals = results['RGB_vals']
    dead_bands = results['dead_bands']
    waveMap = results['waveMap']
    
    # plot stuff
    if bad_data_flag == 1 and gui == 0 and echo_command > 1:
        plt_show(atm_line_index, all_band_removal, band_removal, wavelength, rad_mean, 
                 echo_command, title='Mean Radiance', xlabel='Wavelength (\u03bcm)', ylabel='Radiance (\u03bcf)')
        
        rad_max = np.amax(np.amax(hsi,axis=0), axis=0)
        plt_show(atm_line_index, all_band_removal, band_removal, wavelength, rad_max, 
                 echo_command, title='Max Radiance at Each Band', xlabel='Wavelength (\u03bcm)', ylabel='Radiance (\u03bcf)')
        
        rad_min = np.amin(np.amin(hsi,axis=0), axis=0)
        plt_show(atm_line_index, all_band_removal, band_removal, wavelength, rad_min, 
                 echo_command, title='Min Radiance at Each Band', xlabel='Wavelength (\u03bcm)', ylabel='Radiance (\u03bcf)')               

    return {'atm_line_index':atm_line_index, 'specIndex':specIndex, 'rgb_index':rgb_index, 'hsi':hsi,
            'wavelength':wavelength, 'fwhm':fwhm, 'bad_data_flag':bad_data_flag, 's_atm_line':s_atm_line,
            'e_atm_line':e_atm_line, 'spec_range':spec_range, 'RGB_vals':RGB_vals, 
            'dead_bands':dead_bands, 'waveMap':waveMap, 'rad_mean':rad_mean, 
            'dead_cube_bands':dead_cube_bands, 'waveCoef':waveCoef, 'dead_cube_samples':dead_cube_samples,
            'dead_samples':dead_samples, 'center_wavelength':center_wavelength, 'broad':broad}                                            

def Get_Band_Indices(hsi, waveMap, wavelength, fwhm, band_removal, 
                     hsi_file_suffix, all_band_removal, bad_data_flag, 
                     dead_bands, echo_command):
    """From Get_Band_Indices.py.
    
    Gets atmosphere/false RGB band indices.

    Args:
        hsi (ndarray): HSI data array
        waveMap ([type]): [description]
        wavelength ([type]): [description]
        fwhm ([type]): [description]
        band_removal ([type]): [description]
        hsi_file_suffix ([type]): [description]
        all_band_removal ([type]): [description]
        bad_data_flag ([type]): [description]
        dead_bands ([type]): [description]
        echo_command (int, optional): [description]. 

    Returns:
        [type]: [description]
    """

    #s_atm_line begining of atmosphere lines
    #e_atm_line end of atmosphere lines
    bad_data_flag_loc = 1
    try:
        lam_min_set_loc = wavelength[0]
        lam_max_set_loc = wavelength[wavelength.shape[0]-1]
    except:
        lam_min_set_loc = 20
        lam_max_set_loc = 21
    if (lam_min_set_loc<=0.41 and lam_max_set_loc<=0.6 and lam_max_set_loc>=0.45 and lam_max_set_loc<0.6):
        if echo_command >= 1:
            print("UV")
        spec_range = np.array([0.35, 0.5])
        s_atm_line = np.array([])
        e_atm_line = np.array([])
        RGB_vals = np.array([0.47, 0.44, 0.41])
    elif (lam_min_set_loc<=0.45 and lam_max_set_loc<=1.3 and lam_max_set_loc>=0.8 and lam_max_set_loc<1.2):
        if echo_command >= 1:
            print("VNIR")
        spec_range = np.array([0.40, 1.0])
        s_atm_line = np.array([0.92])
        e_atm_line = np.array([0.98])
        RGB_vals = np.array([0.650, 0.530, 0.480])
    elif (lam_min_set_loc>=0.35 and lam_min_set_loc<0.6 and lam_max_set_loc>=2.2 and lam_max_set_loc<2.6):
        if echo_command >= 1:
            print("VNIRSWIR")
        spec_range = np.array([0.40, 2.48])
        s_atm_line = np.array([0.92, 1.11, 1.35, 1.81])
        e_atm_line = np.array([0.98, 1.16, 1.47, 2.00])
        RGB_vals = np.array([0.650, 0.530, 0.480])
    elif (lam_min_set_loc>=0.45 and lam_min_set_loc<0.6 and lam_max_set_loc>=2.2 and lam_max_set_loc<2.6):
        if echo_command >= 1:
            print("NIRSWIR")
        spec_range = np.array([0.50, 2.48])
        s_atm_line = np.array([0.92, 1.11, 1.35, 1.81])
        e_atm_line = np.array([0.98, 1.16, 1.47, 2.00])
        RGB_vals = np.array([0.750, 0.630, 0.580])
    elif (lam_min_set_loc>=0.89 and lam_min_set_loc<=1.3 and lam_max_set_loc<=2.5 and lam_max_set_loc>=2.0):
        if echo_command >= 1:
            print("SWIR")
        spec_range = np.array([1.0, 2.48])
        s_atm_line = np.array([1.11, 1.35, 1.81])
        e_atm_line = np.array([1.16, 1.47, 2.00])
        RGB_vals = np.array([1.05, 1.3, 1.75])
    elif (lam_min_set_loc>=1.3 and lam_min_set_loc<=1.7 and lam_max_set_loc<=2.5 and lam_max_set_loc>=1.75): 
        if echo_command >= 1:
            print("SWIR I&II")
        spec_range = np.array([1.5, 2.48])
        s_atm_line = np.array([1.78])
        e_atm_line = np.array([2.05])
        RGB_vals = np.array([2.25, 2.06, 1.70])
    elif (lam_min_set_loc>=1.9 and lam_max_set_loc<=2.55 and lam_max_set_loc>2.4 and lam_max_set_loc<2.6): 
        if echo_command >= 1:
            print("SWIR II")
        spec_range = np.array([2.10, 2.485])
        s_atm_line = np.array([])
        e_atm_line = np.array([])
        RGB_vals = np.array([2.42, 2.35, 2.27])
        if lam_min_set_loc>2.3:
                RGB_vals = np.array([2.47, 2.43, 2.39])
        if len(wavelength)>400 and len(wavelength)<=700:
            spec_range = np.array([2.10, 2.48])
            s_atm_line = np.array([2.4723])
            e_atm_line = np.array([2.4724]) 
        if len(wavelength)>700 and len(wavelength)<=1000:
            spec_range = np.array([2.10, 2.48])
            s_atm_line = np.array([2.4158, 2.4186, 2.4348, 2.4458, 2.4502, 2.4625, 2.4719])
            e_atm_line = np.array([2.4169, 2.4202, 2.4373, 2.4485, 2.4524, 2.4636, 2.4738])
        elif len(wavelength)>1000:
            spec_range = np.array([2.10, 2.485])
            s_atm_line = np.array([2.4158, 2.4183, 2.4348, 2.4457, 2.4500, 2.4624, 2.4715])
            e_atm_line = np.array([2.4169, 2.4198, 2.4371, 2.4483, 2.4521, 2.4637, 2.4735])
    elif (lam_min_set_loc<=8.5 and lam_min_set_loc>6.9 and lam_max_set_loc<=14.5 and lam_max_set_loc>10):
        if echo_command >= 1:
            print("LWIR")
        spec_range = np.array([8,13.0])
        s_atm_line = np.array([])
        e_atm_line = np.array([])
        RGB_vals = np.array([11.3, 10.2, 8.6])
    elif (lam_min_set_loc<=3.6 and lam_min_set_loc>3 and lam_max_set_loc<=4.3 and lam_max_set_loc>3.6):
        if echo_command >= 1:
            print("MWIR I")
        spec_range = np.array([3.34, 4.05])
        s_atm_line = np.array([3.667])
        e_atm_line = np.array([3.686])
        RGB_vals = np.array([3.7, 3.5, 3.4])
        if len(wavelength)>1000:
            spec_range = np.array([3.343, 3.750])
            s_atm_line = np.array([3.3453, 3.3542, 3.3628, 3.3712, 3.3796, 3.3900, 3.4030, 3.4145, 3.4270, 3.4624, 3.4715, 3.6744])
            e_atm_line = np.array([3.3477, 3.3589, 3.3693, 3.3723, 3.3810, 3.3930, 3.4056, 3.4183, 3.4300, 3.4637, 3.4735, 3.6780])
    elif (lam_min_set_loc<=4.0 and lam_min_set_loc>3.2 and lam_max_set_loc<=5.6 and lam_max_set_loc>4.7):
        if echo_command >= 1:
            print("MWIR")
        spec_range = np.array([3.34, 5.2])
        s_atm_line = np.array([3.667, 3.90, 4.952, 5.013, 5.075, 5.109, 5.134])
        e_atm_line = np.array([3.686, 4.45, 4.963, 5.033, 5.091, 5.117, 5.157])
        if hsi_file_suffix == ".sc":
            s_atm_line = np.array([4.0])
            e_atm_line = np.array([4.5])
        # RGB_vals = np.array([4.7, 4.0, 3.5])
        RGB_vals = np.array([3.7, 3.5, 3.4])
    elif (lam_min_set_loc<4.7 and lam_min_set_loc>4.0 and lam_max_set_loc<5.6 and lam_max_set_loc>4.7):
        if echo_command >= 1:
            print("MWIR II")
        spec_range = np.array([4.45, 5.30])
        s_atm_line = np.array([4.952, 5.013, 5.075, 5.109, 5.134])
        e_atm_line = np.array([4.963, 5.033, 5.091, 5.117, 5.157])
        RGB_vals = np.array([5.0, 4.75, 4.55])
    else:
        if echo_command >= 1:
            print('Wrong wavelength range')
        bad_data_flag = 0
        
    # make sure spectral range is set
    try:
        spec_range
    except:
        spec_range = []
        RGB_vals = []
        bad_data_flag_loc = 0
    bad_data_flag = bad_data_flag*bad_data_flag_loc
    
    # make spec index for spectral space analysis
    # removes bad or saturated bands etc.
    if band_removal == 1 and all_band_removal == 0 and bad_data_flag == 1:
        specIndex = np.arange(wavelength.shape[0])
        specIndex_c = np.zeros(wavelength.shape[0])
        specStart = np.argmin(abs(wavelength - spec_range[0]))
        specEnd = np.argmin(abs(wavelength - spec_range[1]))
        if (specEnd == len(wavelength)-1):
            specEnd = len(wavelength)+1
        for i in range(len(wavelength)):
            if i >= specStart and i <= specEnd:
                specIndex_c[i]=1
            for j in range(len(s_atm_line)):
                first = np.argmin(np.abs(wavelength-s_atm_line[j]))
                second = np.argmin(np.abs(wavelength-e_atm_line[j]))
                if i >= first and i < second:
                    specIndex_c[i]=0
        keep = specIndex_c>0
        specIndex = specIndex[keep]
        if echo_command >= 1:
            print("Bands inside defined spectral fit regions:", len(specIndex))
        
        # make atmospheric line index
        atm_line_index=[]
        atm_line_index.append(int(specStart))
        for j in range(len(s_atm_line)):
            first = np.argmin(np.abs(wavelength-s_atm_line[j]))-1
            second = np.argmin(np.abs(wavelength-e_atm_line[j]))
            atm_line_index.append(int(first))
            atm_line_index.append(int(second+1))
        if int(specEnd) > specIndex[len(specIndex)-1]:
            atm_line_index.append(int(specIndex[len(specIndex)-1]+1))
        else:
            atm_line_index.append(int(specEnd+1))
        if atm_line_index[len(atm_line_index)-1] == wavelength.shape[0]:
            atm_line_index[len(atm_line_index)-1] = wavelength.shape[0] - 1
    
    # remove atmospheric lines from data
    elif band_removal == 0 and all_band_removal == 1 and bad_data_flag == 1:
        fwhm = list(fwhm)
        for i in range(len(s_atm_line)):
            first = np.argmin(np.abs(wavelength-s_atm_line[i]))
            second = np.argmin(np.abs(wavelength-e_atm_line[i]))
            wavelength = np.delete(wavelength, slice(first, second))
            del fwhm[first:second] 
            new = np.delete(hsi, np.arange(first, second), axis=2)
            hsi = new
            new = np.delete(waveMap, np.arange(first, second), axis=1)
            waveMap = new
        specIndex = np.arange(0,wavelength.shape[0])
        fwhm = np.array(fwhm)
        dead_bands = np.ones(fwhm.shape[0], dtype=bool)
        
        # get indices of atmospehric lines for plotting purposes
        atm_line_index=[]
        atm_line_index.append(0)
        for i in range(len(s_atm_line)):
            first = np.argmin(np.abs(wavelength-s_atm_line[i]))
            second = np.argmin(np.abs(wavelength-e_atm_line[i]))
            atm_line_index.append(first)
            if second>=wavelength.shape[0]:
                second = wavelength.shape[0]-1
            atm_line_index.append(second)
        atm_line_index.append(wavelength.shape[0]-1)
        
        if echo_command >= 1:
            print("Bands after after atmospheric lines removed:", len(specIndex))
    # keep all bands for all algorithms
    else:
        atm_line_index = []
        specIndex = np.arange(0,wavelength.shape[0])
        s_atm_line = []
        e_atm_line = []
    
    # find indices for 3 band  image
    rgb_index=[]
    if bad_data_flag == 1:
        rband = wavelength.tolist().index(min(wavelength, key=lambda x:abs(x-RGB_vals[0])))
        rgb_index.append(rband)
        gband = wavelength.tolist().index(min(wavelength, key=lambda x:abs(x-RGB_vals[1])))
        rgb_index.append(gband)
        bband = wavelength.tolist().index(min(wavelength, key=lambda x:abs(x-RGB_vals[2])))
        rgb_index.append(bband)
        
    return {'atm_line_index':atm_line_index, 'specIndex':specIndex, 'rgb_index':rgb_index,
            'wavelength':wavelength, 'fwhm':fwhm, 'hsi':hsi, 'bad_data_flag':bad_data_flag,
            's_atm_line':s_atm_line, 'e_atm_line':e_atm_line, 'spec_range':spec_range,
            'RGB_vals':RGB_vals, 'dead_bands':dead_bands, 'waveMap':waveMap}

def Read_Envi_Aux(filename, echo_command):
    """ From Read_Envi_Aux.py.

    Args:
        filename (string): aux file name
        echo_command (int): Print results to console (value >=1) or don't (value 0).

    Returns:
        dict: bad pixel/wavelength maps
    """
        
    # define current file and define header
    header = Read_Envi_Header(filename)

    # read in image data, consider interleave and byte order
    if (header.interleave == 'bip' or header.interleave == 'BIP'): odr='C'
    if (header.interleave == 'bil' or header.interleave == 'BIL'): odr='C'
    if (header.interleave == 'bsq' or header.interleave == 'BSQ'): odr='C'
    aux = np.fromfile(filename, dtype=header.type, sep="")
    if hasattr(header, 'byteorder'):
        if header.byteorder == 1:
            aux = aux.byteswap()
            
    # bad data flag
    bad_data_flag = 1
    if np.any(np.isfinite(aux)):
        bad_data_flag = 0
                
    # delete header offset if needed
    if hasattr(header, 'header_offset'):
        if header.header_offset != 0:
            if header.type == 'float64':
                aux = aux[int(header.header_offset/8):int(aux.shape[0])]
            elif header.type == 'float32':
                aux = aux[int(header.header_offset/4):int(aux.shape[0])]
                
    #  reshape based on values from header
    if (header.interleave == 'bip' or header.interleave == 'BIP'):
       aux = aux.reshape([header.lines,header.samples,header.bands], order=odr)
    if (header.interleave == 'bil' or header.interleave == 'BIL'):
       aux = aux.reshape([header.lines,header.bands,header.samples], order=odr)
       aux = np.swapaxes(aux,1,2)
    # this needs to be verified dont have bsq data
    if (header.interleave == 'bsq' or header.interleave == 'BSQ'):
       aux = aux.reshape([header.bands,header.lines,header.samples], order=odr)
       aux = np.swapaxes(aux,0,2)
       aux = np.swapaxes(aux,0,1)
       
    # Mako L2S.dat rotate by 90
    if 'Whisk' in filename:
        aux = aux[::-1,:,:]
       
    # description
    if hasattr(header, 'description'):
        description = header.description
    else:
        description = []
        
    # description
    if hasattr(header, 'band_names'):
        band_names = header.band_names
    else:
        band_names = []
        
    # get aux info
    pyHAT_flag = 0
    for des in description:
        if "pyHAT" in des:
            pyHAT_flag = 1
    if hasattr(header, 'description') and pyHAT_flag == 1:
        if echo_command >= 1:
            print("pyHAT aux file")
        wavelength_map = aux[:,:,0]
        aux_bp_map = aux[:,:,1]
    else:
        if echo_command >= 1:
            print("Other aux file")
        
        if len(band_names)>0:
            if 'BadPixelMap' in band_names:
                aux_bp_map = np.int32(aux[:,:,band_names.index('BadPixelMap')])
            if 'Bad Pixel Map' in band_names:
                aux_bp_map = np.int32(aux[:,:,band_names.index('Bad Pixel Map')])
        else:
            aux_bp_map = np.empty([0])
            
        # get bad pixel map
        wavelength_map = np.empty([0])
        
    return {'aux_bp_map':aux_bp_map, 'wavelength_map':wavelength_map, 'bad_aux_flag':bad_data_flag}

def read_atm(cube_name):
    name,ext = os.path.splitext(cube_name)
    white_name = name+'.white'
    sratm_name = name+'.sratm'
    hratm_name = name+'.hratm'
    mrad_name = name+'.mrad'

    # mean radiance and nesr
    mrad_header = Read_Envi_Header(mrad_name)
    data = np.fromfile(mrad_name, dtype=mrad_header.type)
    if mrad_header.byteorder == 1:
        data = data.byteswap()
    data = data.reshape([mrad_header.lines, mrad_header.samples])
    wavelength = mrad_header.wavelength
    rad_mean = np.array(data[0,:]).flatten()
    NESR_scene = np.array(data[1,:]).flatten()
    
    # whitener
    whitener_header = Read_Envi_Header(white_name)
    data = np.fromfile(white_name, dtype=whitener_header.type)
    if whitener_header.byteorder == 1:
        data = data.byteswap()
    # delete header offset if needed
    if hasattr(whitener_header, 'header_offset'):
        if whitener_header.header_offset != 0:
            data = data[int(whitener_header.header_offset/4):int(data.shape[0])]
    data = data.reshape([whitener_header.lines, whitener_header.samples, whitener_header.bands])
    whitener_rad = data[:,:,0]
    
    # read header
    TEAS_lib_header = Read_Envi_Header(hratm_name)
    # read in  data and reshape based on values from header
    TEAS_lib_hires = np.fromfile(hratm_name, dtype=TEAS_lib_header.type)
    if TEAS_lib_header.byteorder == 1:
        TEAS_lib_hires = TEAS_lib_hires.byteswap()
    TEAS_lib_hires = (TEAS_lib_hires.reshape([TEAS_lib_header.lines, TEAS_lib_header.samples]))
    TEAS_lib_hires.shape
    type(TEAS_lib_hires)
    
    # get Wavelength, Ld, tau, and Lup
    TEAS_wavelength = np.array(TEAS_lib_header.wavelength).flatten()
    downwelling_trans = np.array(TEAS_lib_hires[1,:]).flatten()
    transmission = np.array(TEAS_lib_hires[2,:]).flatten()

    # read header
    TEAS_lib_header = Read_Envi_Header(sratm_name)
    # read in  data and reshape based on values from header
    TEAS_lib_seres = np.fromfile(sratm_name, dtype=TEAS_lib_header.type)
    if TEAS_lib_header.byteorder == 1:
        TEAS_lib_seres = TEAS_lib_seres.byteswap()
    TEAS_lib_seres = (TEAS_lib_seres.reshape([TEAS_lib_header.lines, TEAS_lib_header.samples]))
    TEAS_lib_seres.shape
    type(TEAS_lib_seres)
    
    # get Wavelength, Ld, tau, and Lup
    wavelength_sres = np.array(TEAS_lib_header.wavelength).flatten()
    Lu = np.array(TEAS_lib_seres[0,:]).flatten()
    Lu_np = np.array(TEAS_lib_seres[3,:]).flatten()
    Ld_tau = np.array(TEAS_lib_seres[1,:]).flatten()
    Ld_tau_np = np.array(TEAS_lib_seres[4,:]).flatten()
    if wavelength_sres[0] > 2.6:
        tau = np.array(TEAS_lib_seres[2,:]).flatten()
        tau_np = np.array(TEAS_lib_seres[5,:]).flatten()
    else:
        tau = []
        tau_np = []

    atm_dict = {'hres':{'wavelength':wavelength,'Ld_tau':downwelling_trans,'tau':transmission},
                'sres':{'wavelength':wavelength_sres,'Lu':Lu,'Lu_np':Lu_np,'Ld_tau':Ld_tau,
                       'Ld_tau_np':Ld_tau_np,'tau':tau,'tau_np':tau_np}}
    
    return atm_dict

def plt_show(atm_line_index, all_band_removal, band_removal, wavelength, 
             vector, echo_command, title='', fontsize=14, xlabel='', ylabel=''):
    """ From plt_show.py.

    Args:
        atm_line_index ([type]): [description]
        all_band_removal ([type]): [description]
        band_removal ([type]): [description]
        wavelength ([type]): [description]
        vector ([type]): [description]
        echo_command ([type]): [description]
        title (str, optional): [description]. Defaults to ''.
        fontsize (int, optional): [description]. Defaults to 14.
        xlabel (str, optional): [description]. Defaults to ''.
        ylabel (str, optional): [description]. Defaults to ''.
    """

    plt.figure(dpi=150)
    plt.title(title, fontsize = fontsize, fontweight = 'bold')
    plt.ylabel(ylabel, fontsize = fontsize)
    plt.xlabel(xlabel, fontsize = fontsize)
    if len(atm_line_index)>0:
        if all_band_removal == 0 and band_removal == 1:
            plt.plot(wavelength, vector, color = 'red', linewidth = 0.5)
        for ind in range(int(len(atm_line_index)/2)):
            plt.plot(wavelength[atm_line_index[2*ind]:atm_line_index[2*ind+1]], 
                     vector[atm_line_index[2*ind]:atm_line_index[2*ind+1]], color = 'black', linewidth = 0.5)
    else:
        plt.plot(wavelength, vector, color = 'black', linewidth = 0.5)
    if echo_command > 1:
        plt.show()  

class Read_Envi_Header:
    """
    From Read_Envi_Header.py. 
    
    Reads variables from ENVI header files.
    """

    def __init__(self, filename):
        """Create Envi header reader.

        Args:
            filename (string): Header file is filename+'.hdr' or a file name with extension replaced by '.hdr'.
        """
        # define filename
        self.filename = filename
        # open header file for reading
        headerName = filename+'.hdr'
        if not os.path.exists(headerName):                                        
            headerName = filename.rpartition('.')[0]+'.hdr' 
        headerFile = open(headerName, 'r', encoding = 'utf-8', errors='ignore') 
                
        # initialize variable for reading in wavelengths and spectra names
        start_wvl = False
        start_frmt = False
        start_names = False
        start_class_names = False
        start_sigma = False
        start_fwhm = False
        first_lines = True
        first_bands = True
        start_wavecoef = False
        start_broadcoef = False
        first_samples = True
        first_date = True
        first_byte = True
        first_header_offset = True
        start_description = False
        start_band_names = False
        start_fwhm = False
        start_removed_bands = False
        start_removed_samples = False
        self.header_offset = 0
        
        # loop through lines in the file searching for strings
        for num, line in enumerate(headerFile, 1):
            # define variables for samples, lines, bands, and interleave
            if first_header_offset:
                if "header offset" in line: 
                    self.header_offset=line.split("=")[1]
                    first_header_offset = False
            if first_header_offset:
                if "HEADER OFFSET" in line: 
                    self.header_offset=line.split("=")[1]
                    first_header_offset = False
            if first_samples:
                if "samples" in line: 
                    self.samples=line.split("=")[1]
                    first_samples = False
            if first_samples:
                if "SAMPLES" in line: 
                    self.samples=line.split("=")[1]
                    first_samples = False
            if "removed samples =" in line and "{" in line:
                start_removed_samples = True
                self.removed_samples = []   
            if first_lines:            
                if "lines" in line:
                    self.lines=line.split("=")[1]
                    first_lines = False
            if first_lines:            
                if "LINES" in line:
                    self.lines=line.split("=")[1]
                    first_lines = False
            if first_bands:
                if "bands" in line: 
                    self.bands=line.split("=")[1]
                    first_bands = False
            if first_bands:
                if "BANDS" in line: 
                    self.bands=line.split("=")[1]
                    first_bands = False
            if "removed bands =" in line and "{" in line:
                start_removed_bands = True
                self.removed_bands = []  
            if "interleave" in line: self.interleave=line.split("=")[1]
            if "INTERLEAVE" in line: self.interleave=line.split("=")[1]
            if "data type" in line: self.type=line.split("=")[1]
            if "DATA TYPE" in line: self.type=line.split("=")[1]
            if "sensor type" in line: self.sensor_type=line.split("=")[1]
            if "SENSOR TYPE" in line: self.sensor_type=line.split("=")[1]
            if first_date:
                if "date" in line and "=" in line: 
                    self.date=line.split("=")[1]
                    first_date = False
            if first_byte:
                if "byte order" in line:
                    self.byteorder=line.split("=")[1]
                    first_byte = False
            if first_byte:
                if "BYTE ORDER" in line:
                    self.byteorder=line.split("=")[1]
                    first_byte = False
            if "sceneAltitude" in line: self.groundEl=line.split("=")[1]
            
            # aerospace style .hdr
            if "target latitude" in line: self.target_latitude=line.split("=")[1]
            if "target longitude" in line: self.target_longitude=line.split("=")[1]
            if "plane HAE (m)" in line: self.plane_HAE=line.split("=")[1]
            if "target HAE (m)" in line: self.target_HAE=line.split("=")[1]
            if "target standoff (km)" in line: self.target_standoff=line.split("=")[1]
            
            # aces-hy header
            if "center latitude" in line: self.target_latitude=line.split("=")[1]
            if "center longitude" in line: self.target_longitude=line.split("=")[1]
            if "sensor altitude" in line: self.plane_HAE=line.split("=")[1]
            if "ground height" in line: self.target_HAE=line.split("=")[1]
            
            # start adding lines to wvl once "wavelength" is found
            if "wavelength =" in line and "{" in line:
                start_wvl = True
                self.wvl = []
            elif "WAVELENGTH =" in line and "{" in line:
                start_wvl = True
                self.wvl = []
            # start adding lines to frmt once "frame time" is found
            elif "frame time =" in line and "{" in line:
                start_frmt = True
                self.frmt = []
            # start adding lines to names once "spectra names" is found                
            elif "spectra names =" in line and "{" in line:
                start_names = True
                self.names = []
            # start adding lines to class once "class names" is found                
            elif "class names =" in line and "{" in line:
                start_class_names = True
                self.class_names = []
            # start adding lines to sigma once "sigma" is found                
            elif "sigma =" in line and "{" in line:
                start_sigma = True
                self.sigma = []
            # start adding lines to fwhm once "fwhm" is found             
            elif "fwhm =" in line and "{" in line:
                start_fwhm = True
                self.fwhm = []  
            # start adding lines to wavecoef once "wavecoef" is found  
            elif "wavecoef =" in line and "{" in line:
                start_wavecoef = True
                self.wavecoef = []
                # start adding lines to broadcoef once "broadcoef" is found  
            elif "broadcoef =" in line and "{" in line:
                start_broadcoef = True
                self.broadcoef = []
            # start adding lines to description once "description" is found  
            elif "description =" in line and "{" in line:
                start_description = True
                self.description = []
            elif "DESCRIPTION =" in line and "{" in line:
                start_description = True
                self.description = []
            # start adding lines to band names once "description" is found  
            elif "band names =" in line and "{" in line:
                start_band_names = True
                self.band_names = []
            # start adding lines to band names once "description" is found  
            elif "BAND NAMES =" in line and "{" in line:
                start_band_names = True
                self.band_names = []
                                
            # if key word has been found, append lines
            if start_wvl: self.wvl.append(line)
            if start_frmt: self.frmt.append(line)
            if start_names: self.names.append(line)
            if start_class_names: self.class_names.append(line)
            if start_sigma: self.sigma.append(line)
            if start_fwhm: self.fwhm.append(line)
            if start_wavecoef: self.wavecoef.append(line)
            if start_broadcoef: self.broadcoef.append(line)
            if start_description: self.description.append(line)
            if start_band_names: self.band_names.append(line)
            if start_removed_bands: self.removed_bands.append(line)
            if start_removed_samples: self.removed_samples.append(line)
                
            # stop adding lines next } is found
            if start_wvl and "}" in line: start_wvl = False
            if start_frmt and "}" in line: start_frmt = False
            if start_names and "}" in line: start_names = False
            if start_class_names and "}" in line: start_class_names = False
            if start_sigma and "}" in line: start_sigma = False
            if start_fwhm and "}" in line: start_fwhm = False
            if start_wavecoef and "}" in line: start_wavecoef = False
            if start_broadcoef and "}" in line: start_broadcoef = False
            if start_description and "}" in line: start_description = False
            if start_band_names and "}" in line: start_band_names = False
            if start_removed_bands and "}" in line: start_removed_bands = False
            if start_removed_samples and "}" in line: start_removed_samples = False
                
        # close header file
        headerFile.close()

        # initialize new variable for wavelength array if wvl exists
        # extract lines from wvl, split values and convert to float
        match_number = re.compile('-?\ *[0-9]+\.?[0-9]*(?:[Ee]\ *-?\ *[0-9]+)?')
        
        if hasattr(self, 'description'):
            self.description = [des.replace('\n', '') for des in self.description]
            self.description = [des.replace('}', '') for des in self.description]
            self.description = [des.replace('{', '') for des in self.description]

        if hasattr(self, 'wvl'): 
            self.wavelength = []
            temp = ''
            for i in range(len(self.wvl)):
                temp = temp+self.wvl[i]
            dum = temp.split(",")
            for i in range(len(dum)):
                numbers = [float(x) for x in re.findall(match_number, dum[i])]
                dum[i] = numbers[0]
            dum = np.array(dum)
            self.wavelength = np.array([float(x) for x in dum if x != ''],
                    dtype="float32")
            
        if hasattr(self, 'frmt'): 
            self.frame_time = []
            temp = ''
            for i in range(len(self.frmt)):
                temp = temp+self.frmt[i]
            dum = temp.split(",")
            dum[0] = dum[0].strip('frame time = {')
            for i in range(len(dum)):
               dum[i] = dum[i].strip('\n')
               dum[i] = dum[i].strip('}')
            self.frame_time = np.array(dum)

        # extract lines from sigma, split values and convert to float
        if hasattr(self, 'sigma'): 
            temp = ''
            for i in range(len(self.sigma)):
                temp = temp+self.sigma[i]
            dum = temp.split(",")
            for i in range(len(dum)):
                numbers = [float(x) for x in re.findall(match_number, dum[i])]
                dum[i] = numbers[0]
            dum = np.array(dum)
            self.sigma = np.array([float(x) for x in dum if x != ''],
                    dtype="float32")

        # extract lines from removed_bands, split values and convert to int
        if hasattr(self, 'removed_bands'):
            temp = ''
            for i in range(len(self.removed_bands)):
                temp = temp+self.removed_bands[i]
            dum = temp.split(",")
            for i in range(len(dum)):
                numbers = [float(x) for x in re.findall(match_number, dum[i])]
                if len(numbers) == 2:
                    numbers = float(numbers[0])*10**int(numbers[1])
                    dum[i] = numbers
                else:
                    dum[i] = numbers[0]
            dum = np.array(dum)
            self.removed_bands = np.array([float(x) for x in dum if x != ''],
                    dtype="int32") 
            
        # extract lines from removed_samples, split values and convert to int
        if hasattr(self, 'removed_samples'):
            temp = ''
            for i in range(len(self.removed_samples)):
                temp = temp+self.removed_samples[i]
            dum = temp.split(",")
            for i in range(len(dum)):
                numbers = [float(x) for x in re.findall(match_number, dum[i])]
                if len(numbers) == 2:
                    numbers = float(numbers[0])*10**int(numbers[1])
                    dum[i] = numbers
                else:
                    dum[i] = numbers[0]
            dum = np.array(dum)
            self.removed_samples = np.array([float(x) for x in dum if x != ''],
                    dtype="int32")
            
        # extract lines from fwhm, split values and convert to float
        if hasattr(self, 'fwhm'): 
            temp = ''
            for i in range(len(self.fwhm)):
                temp = temp+self.fwhm[i]
            dum = temp.split(",")
            for i in range(len(dum)):
                numbers = [float(x) for x in re.findall(match_number, dum[i])]
                dum[i] = numbers[0]
            dum = np.array(dum)
            self.fwhm = np.array([float(x) for x in dum if x != ''],
                    dtype="float32")
            
        # extract lines from wavecoef, split values and convert to float
        if hasattr(self, 'wavecoef'):
            temp = ''
            for i in range(len(self.wavecoef)):
                temp = temp+self.wavecoef[i]
            dum = temp.split(",")
            for i in range(len(dum)):
                numbers = [float(x) for x in re.findall(match_number, dum[i])]
                if len(numbers) == 2:
                    numbers = float(numbers[0])*10**int(numbers[1])
                    dum[i] = numbers
                else:
                    dum[i] = numbers[0]
            dum = np.array(dum)
            self.wavecoef = np.array([float(x) for x in dum if x != ''],
                    dtype="float32") 
            
        # extract lines from broadcoef, split values and convert to float
        if hasattr(self, 'broadcoef'):
            temp = ''
            for i in range(len(self.broadcoef)):
                temp = temp+self.broadcoef[i]
            dum = temp.split(",")
            for i in range(len(dum)):
                numbers = [float(x) for x in re.findall(match_number, dum[i])]
                dum[i] = numbers[0]
            dum = np.array(dum)
            self.broadcoef = np.array([float(x) for x in dum if x != ''],
                    dtype="float32") 
            
        # extract lines and split values for spectra names if spectra names exists
        if hasattr(self, 'names'):
            temp = ''
            for i in range(len(self.names)):
                temp = temp+self.names[i]
            temp = temp[17:len(temp)]
            try:
                dum = pp.commaSeparatedList.parseString(temp).asList()
            except:
                dum = pp.pyparsing_common.comma_separated_list.parseString(temp).asList()
            for i in range(len(dum)):
                dum[i] = re.sub('[}\n]', '', dum[i])
                dum[i] = dum[i].lstrip()
            self.names = [x for x in dum if x != '']
        
        # extract lines and split values for class names if classes exists
        if hasattr(self, 'class_names'):
            temp = ''
            for i in range(len(self.class_names)):
                temp = temp+self.class_names[i]
            temp = temp[16:len(temp)]
            try:
                dum = pp.commaSeparatedList.parseString(temp).asList()
            except:
                dum = pp.pyparsing_common.comma_separated_list.parseString(temp).asList()
            for i in range(len(dum)):
                dum[i] = re.sub('[}\n]', '', dum[i])
                dum[i] = dum[i].lstrip()
            self.class_names = [x for x in dum if x != '']
            
        # extract lines and split values for class names if classes exists
        if hasattr(self, 'band_names'):
            temp = ''
            for i in range(len(self.band_names)):
                temp = temp+self.band_names[i]
            temp = temp[14:len(temp)]
            try:
                dum = pp.commaSeparatedList.parseString(temp).asList()
            except:
                dum = pp.pyparsing_common.comma_separated_list.parseString(temp).asList()
            for i in range(len(dum)):
                dum[i] = re.sub('[}\n]', '', dum[i])
                dum[i] = dum[i].lstrip()
            self.band_names = [x for x in dum if x != '']
            
        # convert values from strings to decimals
        if hasattr(self, 'lines'):  self.lines = int(self.lines)
        if hasattr(self, 'samples'): self.samples = int(self.samples)
        try:
            if hasattr(self, 'bands'):  self.bands = int(self.bands)
        except:
            if hasattr(self, 'bands'):  self.bands = int(-999)
        if hasattr(self, 'groundEl'): self.groundEl = float(self.groundEl)
        if hasattr(self, 'header_offset'): self.header_offset = int(self.header_offset)
        if hasattr(self, 'interleave'): 
            self.interleave = self.interleave.split("\n")[0]
            self.interleave = self.interleave.replace(" ","")
        if hasattr(self, 'type'): 
            self.type = self.type.split("\n")[0]
            self.type = self.type.replace(" ","")
            self.type = envi_to_python_data_type(self.type)
        if hasattr(self, 'sensor_type'): 
            self.sensor_type = self.sensor_type.split("\n")[0]
            self.sensor_type = self.sensor_type.replace(" ","")
        if hasattr(self, 'date'):
            temp = self.date.split(" ")[1]
            if "/" in temp:
                self.year = temp.split("/")[0]
                self.month = temp.split("/")[1]
                self.day = temp.split("/")[2]
        if hasattr(self, 'byteorder'):
            self.byteorder = int(self.byteorder)

def Read_Whitener(white_name):
    # whitener
    whitener_header = Read_Envi_Header(white_name)
    data = np.fromfile(white_name, dtype=whitener_header.type)
    if whitener_header.byteorder == 1:
        data = data.byteswap()
    # delete header offset if needed
    
    if hasattr(whitener_header, 'header_offset'):
        if whitener_header.header_offset != 0:
            data = data[int(whitener_header.header_offset/4):int(data.shape[0])]
    
    data = data.reshape([whitener_header.lines, whitener_header.samples, whitener_header.bands])
    whitener_rad = data[:,:,0]

    return whitener_rad

def Read_Mean(mrad_name):
    # mean radiance and nesr
    mrad_header = Read_Envi_Header(mrad_name)
    data = np.fromfile(mrad_name, dtype=mrad_header.type,offset=mrad_header.header_offset)
    if mrad_header.byteorder == 1:
        data = data.byteswap()
    data = data.reshape([mrad_header.lines, mrad_header.samples])
    wavelength = mrad_header.wavelength
    rad_mean = np.array(data[0,:]).flatten()
    NESR_scene = np.array(data[1,:]).flatten()

    return (rad_mean,NESR_scene)

# function to convert envi data types to python data types
def envi_to_python_data_type(argument):
    switcher = {
        '1': "uint8",
        '2': "int16",
        '4': "float32",
        '5': "float64",
        '8': "float16",
        '12': "uint16",
    }
    return switcher.get(argument)
