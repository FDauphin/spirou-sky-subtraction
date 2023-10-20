"""Fits hydroxyl lines measured in Rousselot et al. (2000) to SPIRou sky spectra.

This script reads in a .txt file of hydroxyl lines and SPIRou sky spectra .fits
files. It loops through each file, records various metadata from the .fits file,
fits every line from the .txt file, records fitting metrics, and saves all the
data into a .csv file. It defaults to normalizing the spectra before fitting,
fitting a double Gaussian profile with a linear component, and disables plotting.
Changing these parameters disables normalization, fits to different profiles,
and enables plotting of the fit.

Notes
-----
- The different types of flux models that can be fit are singe or double profiles 
of Gaussian or Lorentzian functions with or without a linear component (8 models
in total). 
- The metadata recorded include exposure time, MJD, airmass, medians
across the whole spectrum, J/H/K bands, and the ratio between K and J band sums.
- The measurements recorded include local spectrum median, local spectrum median 
absolute deviation, sum from the flux model, the fitted parameters such as 
amplitude, line center (mu), line width (sigma; and its variance), slope, 
intercept, fitting metrics such as relative error, mean squared error, Pearson
correlation, Spearman correlation, p-value of the Spearman correlation, and peak
ratio which determines if a fitted profile is actually background.
- The script also predetermines initial fitting parameters and parameter bounds
for improved optimization.
- 

Use
---
    This script can be run from the command line:

        >>> python fit_lines.py [data_dir] [save_path]

    The user must provide two arguments: data_dir and save_path. The former is
    the global path to the directory that contains all the SPIRou sky spectra.
    The latter is the path to save the output file.

    This script can also be imported in a jupyter notebook:

        >>> from fit_lines import fit_line

"""

import os
import sys
from glob import glob
from tqdm import tqdm

import numpy as np
from matplotlib import pyplot as plt
import pandas as pd
from scipy.integrate import simpson
from scipy.optimize import curve_fit
from scipy.signal import argrelextrema
from scipy.stats import spearmanr, median_abs_deviation

from astropy.io import fits
from astropy.convolution import convolve, Gaussian1DKernel
from astropy.stats import sigma_clipped_stats

METHODS = ['gauss'
           'gauss linear',
           'gauss double',
           'gauss double linear',
           'lorentz',
           'lorentz linear',
           'lorentz double',
           'lorentz double linear']

def rename_columns(df):
    new_cols = {}
    for i in df.columns:
        try:
            num, metric = i.split('_')
            new_num = round(float(num), 2)
            new_i = f'{new_num}_{metric}'
            new_cols[i] = new_i
        except:
            new_cols[i] = i
    df = df.rename(columns=new_cols)
    return df

def func_gauss(x, amp, mu, sigma):
    """ Gaussian function.

    Parameters
    ----------
    x : array
        The x values.
    amp : float
        The amplitude.
    mu : float
        The center of the function.
    sigma : float
        The spread of the function.

    Returns
    -------
    y : array
        The y values.
    """

    y = amp * np.sqrt(2*np.pi*sigma**2) ** -1 * \
        np.exp(-(x-mu)**2 / (2*sigma**2))

    return y

def func_gauss_linear(x, amp, mu, sigma, m, b):
    """ Gaussian function with a linear term.

    Parameters
    ----------
    x : array
        The x values.
    amp : float
        The amplitude.
    mu : float
        The center of the function.
    sigma : float
        The spread of the function.
    m : float
        The slope.
    b : float
        The y-intercept.

    Returns
    -------
    y : array
        The y values.
    """

    y = func_gauss(x, amp, mu, sigma) + m * x + b
    return y

def func_gauss_double(x, amp1, mu1, sigma1, amp2, mu2, sigma2):
    """ Double Gaussian function.

    Parameters
    ----------
    x : array
        The x values.
    amp1 : float
        The amplitude of the first gaussian.
    mu1 : float
        The center of the first gaussian.
    sigma1 : float
        The spread of the first gaussian.
    amp2 : float
        The amplitude of the second gaussian.
    mu2 : float
        The center of the second gaussian.
    sigma2 : float
        The spread of the second gaussian.

    Returns
    -------
    y : array
        The y values.
    """

    y = func_gauss(x, amp1, mu1, sigma1) + func_gauss(x, amp2, mu2, sigma2)
    return y

def func_gauss_double_linear(x, amp1, mu1, sigma1, amp2, mu2, sigma2, m, b):
    """ Double Gaussian function with a linear term.

    Parameters
    ----------
    x : array
        The x values.
    amp1 : float
        The amplitude of the first gaussian.
    mu1 : float
        The center of the first gaussian.
    sigma1 : float
        The spread of the first gaussian.
    amp2 : float
        The amplitude of the second gaussian.
    mu2 : float
        The center of the second gaussian.
    sigma2 : float
        The spread of the second gaussian.
    m : float
        The slope.
    b : float
        The y-intercept.

    Returns
    -------
    y : array
        The y values.
    """

    y = (func_gauss_linear(x, amp1, mu1, sigma1, m, b) + \
         func_gauss(x, amp2, mu2, sigma2))
    return y

def func_lorentz(x, amp, x_0, hwhm):
    """ Lorentz/Cauchy function.

    Parameters
    ----------
    x : array
        The x values.
    amp : float
        The amplitude.
    x_0 : float
        The location parameter.
    hwhm : float
        The half-width half-maximum (scale parameter).

    Returns
    -------
    y : array
        The y values.
    """

    y = amp * (np.pi * hwhm * (1 + (x-x_0)**2 / hwhm**2)) ** -1
    return y

def func_lorentz_linear(x, amp, x_0, hwhm, m, b):
    """ Lorentz/Cauchy function with a linear term.

    Parameters
    ----------
    x : array
        The x values.
    amp : float
        The amplitude.
    x_0 : float
        The location parameter.
    hwhm : float
        The half-width half-maximum (scale parameter).
    m : float
        The slope.
    b : float
        The y-intercept.

    Returns
    -------
    y : array
        The y values.
    """

    y = func_lorentz(x, amp, x_0, hwhm) + m * x + b
    return y

def func_lorentz_double(x, amp1, x_01, hwhm1, amp2, x_02, hwhm2):
    """ Double Lorentz/Cauchy function.

    Parameters
    ----------
    x : array
        The x values.
    amp1 : float
        The amplitude of the first Lorentzian.
    x_01 : float
        The location parameter of the first Lorentzian.
    hwhm1 : float
        The half-width half-maximum (scale parameter) of the first Lorentzian.
    amp2 : float
        The amplitude of the second Lorentzian.
    x_02 : float
        The location parameter of the second Lorentzian.
    hwhm2 : float
        The half-width half-maximum (scale parameter) of the second Lorentzian.

    Returns
    -------
    y : array
        The y values.
    """

    y = func_lorentz(x, amp1, x_01, hwhm1) + func_lorentz(x, amp2, x_02, hwhm2)
    return y

def func_lorentz_double_linear(x, amp1, x_01, hwhm1, amp2, x_02, hwhm2, m, b):
    """ Double Lorentz/Cauchy function with a linear term.

    Parameters
    ----------
    x : array
        The x values.
    amp1 : float
        The amplitude of the first Lorentzian.
    x_01 : float
        The location parameter of the first Lorentzian.
    hwhm1 : float
        The half-width half-maximum (scale parameter) of the first Lorentzian.
    amp2 : float
        The amplitude of the second Lorentzian.
    x_02 : float
        The location parameter of the second Lorentzian.
    hwhm2 : float
        The half-width half-maximum (scale parameter) of the second Lorentzian.
    m : float
        The slope.
    b : float
        The y-intercept.

    Returns
    -------
    y : array
        The y values.
    """

    y = (func_lorentz_linear(x, amp1, x_01, hwhm1, m, b) + \
         func_lorentz(x, amp2, x_02, hwhm2))
    return y

def make_dict(lines, method='gauss'):
    """ Make dictionary for metadata and fits for specific spectral lines.

    The dictionary can be converted to a dataframe.

    Parameters
    ----------
    lines : list of floats
        The spectral lines in Angstroms to be fit.
    method : str, default='gauss'
        The fitting function used: single or double Gaussian/Lorentzian
        with/without a linear term.

    Returns
    -------
    df_dict : dict of lists
        A dictionary with empty lists to be appended when fitting.
    """

    col = ['file', 'spec_exptime', 'spec_mjd', 'spec_median', 'spec_median_j',
           'spec_median_h', 'spec_median_k', 'spec_k_over_j', 'airmass']

    for i in lines:
        col.append(str(i) + '_median')
        col.append(str(i) + '_mad')
        col.append(str(i) + '_sum')
        col.append(str(i) + '_amp1')
        col.append(str(i) + '_mu1')
        col.append(str(i) + '_sigma1')
        col.append(str(i) + '_sigma1_cov')
        col.append(str(i) + '_error')
        col.append(str(i) + '_mse')
        col.append(str(i) + '_pearson')
        col.append(str(i) + '_spearman')
        col.append(str(i) + '_pvalue')
        if 'linear' in method:
            col.append(str(i) + '_slope')
            col.append(str(i) + '_intercept')
        if 'double' in method:
            col.append(str(i) + '_amp2')
            col.append(str(i) + '_mu2')
            col.append(str(i) + '_sigma2')
            col.append(str(i) + '_sigma2_cov')
            col.append(str(i) + '_peak_ratio')

    df_dict = {}
    for i in col:
        df_dict[i] = []

    return df_dict

def get_wfe_metadata(file, normalize=True, j=[11700.0, 13700.0], 
                     h=[14700.0, 17000.0], k=[19500.0, 23500.0]):
    """Retrieve the wavelengths, fluxes, and metadata for the given sky specta.

    The metadata recorded are file name, exposure time, observation date, 
    airmass, the spectrum's various medians (whole spectra, J band, H band, 
    K band) and the ratio of K band flux to J band flux.

    Parameters
    ----------
    file : str
        File name
    normalize : bool, default=True
        If True, normalize flux by median.
    j : list of floats, default=[11700.0, 13700.0]
        The minimum and maximum values for the J band in Angstroms.
        In theory, this can be replaced with any values.
    h : list of floats, default=[14700.0, 17000.0]
        The minimum and maximum values for the H band in Angstroms.
        In theory, this can be replaced with any values.
    k : list of floats, default=[19500.0, 23500.0]
        The minimum and maximum values for the K band in Angstroms.
        In theory, this can be replaced with any values.

    Returns
    -------
    wave : array
        Wavelengths measured in Angstroms. Indices are aligned with flux.
    flux : array
        Fluxes measured at the corresponding wavelength.
    df_dict_spec : dict
        Recorded spectra metadata.
    """

    df_dict_spec = {}

    # Open file and data
    hdu = fits.open(file)[1]
    hdr = hdu.header
    try:
        exptime = hdr['exptime']
        mjd_start = hdr['mjd-obs']
        mjd_end = hdr['mjdend']
        mjd = (mjd_start + mjd_end) / 2
        airmass = hdr['airmass']
    except KeyError:
        exptime = fits.getval(file, 'exptime')
        mjd = fits.getval(file, 'mjdmid')
        airmass = fits.getval(file, 'airmass')

    sky_spec = hdu.data
    wave = sky_spec.field(0)
    
    # Convert to Angstroms (optimal for fitting)
    wave = wave * 10

    # Normalize flux by exptime and median
    flux = sky_spec.field(1) / exptime
    flux_med = np.nanmedian(flux)
    if normalize:
        flux = flux / flux_med

    #Calculate various band medians
    mask_j = (wave >= j[0]) & (wave <= j[1])
    mask_h = (wave >= h[0]) & (wave <= h[1])
    mask_k = (wave >= k[0]) & (wave <= k[1])
    # Future: clipped median (negligible changes)
    flux_j_med = np.nanmedian(flux[mask_j])
    flux_h_med = np.nanmedian(flux[mask_h])
    flux_k_med = np.nanmedian(flux[mask_k])

    # sum(K band) / sum(J band)
    k_j_sum = np.nansum(flux[mask_k]) / np.nansum(flux[mask_j])

    # Record metadata
    df_dict_spec['file'] = os.path.split(file)[1]
    df_dict_spec['spec_exptime'] = exptime
    df_dict_spec['spec_mjd'] = mjd
    df_dict_spec['spec_median'] = flux_med
    df_dict_spec['spec_median_j'] = flux_j_med
    df_dict_spec['spec_median_h'] = flux_h_med
    df_dict_spec['spec_median_k'] = flux_k_med
    df_dict_spec['spec_k_over_j'] = k_j_sum
    df_dict_spec['airmass'] = airmass

    return wave, flux, df_dict_spec

def find_initial_amp_mu(line, wave_local, flux_local, stddev=2):
    """ Find the initial guesses for amplitude and mu.

    Convolve the flux with a Gaussian filter to smooth out low signal to noise
    relative maxima. The relative maximum closest to the reference line
    determines the first set of initial parameters. The higher local maximum
    adjacent to the first maximum determines the second set of initial 
    parameters.

    Parameters
    ----------
    line : float
        Spectral line to be fit in Angstroms.
    wave_local : array
        Wavelengths in nm centered around line +/- a predetermined width.
    flux_local : array
        Flux centered around line +/- a predetermined width.
    stddev : float, default=2
        Standard deviation of the Gaussian filter. If stddev is 0, a filter
        isn't used.

    Returns
    -------
    amp_init : array
        The initial amplitude values for fitting.
    mu_init : array
        The initial mu values for fitting.
    """

    # Convolve flux to remove low SNR relative maxima
    if stddev==0:
        flux_conv = flux_local
    else:
        kernel = Gaussian1DKernel(stddev=stddev)
        flux_conv = convolve(flux_local, kernel)

    # Find local maxima of convolved flux
    inds_local_flux_maxima = argrelextrema(flux_conv, np.greater)
    wave_cands = wave_local[inds_local_flux_maxima]
    flux_cands = flux_local[inds_local_flux_maxima]

    # Find local maxima closest to spectral line
    abs_diff_wave_line = np.abs(wave_cands - line)
    inds = np.argsort(abs_diff_wave_line)
    ind_init_1 = inds[0]
    amp_init_1 = flux_cands[ind_init_1]
    mu_init_1 = wave_cands[ind_init_1]

    inds_length = inds.shape[0]
    
    # If there is only one max, then the second initial guess is the first
    if inds_length == 1:
        ind_init_2 = ind_init_1

    # If there are only two maxes, define the second guess
    if inds_length == 2:
        ind_init_2 = inds[1]

    # If there are more than two maxes, choose the largest max next to mu_init_1
    if inds_length > 2:
        # If the global max is leftmost, choose directly right as the next max
        if ind_init_1 == 0:
            ind_init_2 = 1
        # If the global max is rightmost, choose directly left as the next max
        elif ind_init_1 == inds_length - 1:
            ind_init_2 = inds_length - 2
        # If not, then choose the largest from either left or right
        else:
            ind_init_left = ind_init_1 - 1
            ind_init_right = ind_init_1 + 1
            flux_left = flux_cands[ind_init_left]
            flux_right = flux_cands[ind_init_right]
            if flux_left > flux_right:
                ind_init_2 = ind_init_left
            else:
                ind_init_2 = ind_init_right

    # Define second set of inital guesses for amp and mu
    amp_init_2 = flux_cands[ind_init_2]
    mu_init_2 = wave_cands[ind_init_2]

    # Pack initial parameters
    amp_init = np.array([amp_init_1, amp_init_2])
    mu_init = np.array([mu_init_1, mu_init_2])

    return amp_init, mu_init

def initial_values_from_method(method, p0_init):
    """ Choose the initial parameters and function to fit the lines.

    Parameters
    ----------
    method : str, default='gauss'
        The fitting function used: single or double Gaussian/Lorentzian
        with/without a linear term.
    p0_init : array
        The initial values used for fitting. The array's elements are:
        amp_init_1, line_init_1, sigma_init, amp_init_2, line_init_2,
        sigma_init, m_init, and b_init.

    Returns
    -------
    func : function
        The function used for fitting.
    p0 : array
        The initial values based on the method.
    """

    if method in 'gauss':
        func = func_gauss
        p0_ind = [0, 1, 2]
    elif method == 'gauss linear':
        func = func_gauss_linear
        p0_ind = [0, 1, 2, 6, 7]
    elif method == 'gauss double':
        func = func_gauss_double
        p0_ind = [0, 1, 2, 3, 4, 5]
    elif method == 'gauss double linear':
        func = func_gauss_double_linear
        p0_ind = [0, 1, 2, 3, 4, 5, 6, 7]
    elif method in 'lorentz':
        func = func_lorentz
        p0_ind = [0, 1, 2]
    elif method == 'lorentz linear':
        func = func_lorentz_linear
        p0_ind = [0, 1, 2, 6, 7]
    elif method == 'lorentz double':
        func = func_lorentz_double
        p0_ind = [0, 1, 2, 3, 4, 5]
    elif method == 'lorentz double linear':
        func = func_lorentz_double_linear
        p0_ind = [0, 1, 2, 3, 4, 5, 6, 7]
    else:
        msg = f'{method} is not an acceptable method: choose from {METHODS}'
        raise NameError(msg)

    p0 = p0_init[p0_ind]
    return func, p0

def find_parameter_bounds(line, flux_local, width=1.5, bounds_lower_sigma=0,
                          bounds_upper_sigma=10):
    """ Create a tuple of boundaries for model parameters when fitting using
    curve_fit.

    Boundaries restrict extreme values from being fit. For amplitude, the
    if-else statements decrease or increase the bounds based on the global
    minimum and maximum. The parameter mu is bounded to twice the width of
    the local spectrum. Sigma should be positive, and ideally lower than 10
    Angstroms. Since the fits are relatively far from the origin 
    (~thousands of Angstroms), the slopes and intercepts tend to be on the same 
    order of magnitude so the domain for those parameters are all real numbers.

    Parameters
    ----------
    line : float
        Spectral line in Angstroms to be fit.
    flux_local : array
        Flux centered around line +/- a predetermined width.
    width : float, default=1.5
        The width of the local spectra centered about the line, i.e. the min
        and max wavelengths in Angstroms used when fitting is 
        [line - width, line + width]. A width of 1.5 usually contains around 
        40-90 data points.
    bounds_lower_sigma : float, default=0
        Lower bound for sigma parameter in Angstroms.
    bounds_upper_sigma : float, default=10
        Upper bound for sigma parameter in Angstroms.

    Returns
    -------
    bounds : tuple of lists
        Lower and upper bounds for fitting parameters: amp1, mu1, sigma1,
        amp2, mu2, sigma2, slope, and intercept.
    """

    # Find lower bound for amplitudes
    bounds_lower_amp = np.nanmin(flux_local)
    if bounds_lower_amp > 0:
        bounds_lower_amp /= 10
    else:
        bounds_lower_amp *= 10

    # Find upper bound for amplitudes
    bounds_upper_amp = np.nanmax(flux_local)
    if bounds_upper_amp > 0:
        bounds_upper_amp *= 10
    else:
        bounds_upper_amp /= 10

    # Define lower/upper mu bounds as twice the width of the spectrum local
    bounds_lower_mu = line - width * 2
    bounds_upper_mu = line + width * 2

    # Define lower/upper bounds for slope/intercept as +/- infinity
    bounds_lower_slope = -np.inf
    bounds_upper_slope = np.inf
    bounds_lower_intercept = -np.inf
    bounds_upper_intercept = np.inf

    # Create bounds tuple for curve fit
    bounds_lower = [bounds_lower_amp, bounds_lower_mu, bounds_lower_sigma,
                    bounds_lower_amp, bounds_lower_mu, bounds_lower_sigma,
                    bounds_lower_slope, bounds_lower_intercept]
    bounds_upper = [bounds_upper_amp, bounds_upper_mu, bounds_upper_sigma,
                    bounds_upper_amp, bounds_upper_mu, bounds_upper_sigma,
                    bounds_upper_slope, bounds_upper_intercept]

    bounds = (bounds_lower, bounds_upper)

    return bounds

def fit_line(line, wave, flux, width=1.5, sigma_init=0.1, method='gauss',
             plot=False):
    """ Fit a spectral line.

    First, grab a local spectrum centered on the line within a specified
    width. Second, fit the line to the given function method. Third, integrate
    the model fit with the continuum subtracted using Simpson's rule. Forth,
    record best fit parameters and evaluation metrics of the fit.
    Finally, plot the data if necessary.

    The metadata recorded are the local spectra median and integrated flux. The
    best fit parameters recorded are the amplitudes, means, and standard
    deviations of a single or double distribution, in addition to the slope,
    and y-intercept. The metrics recorded are the percentage error, the mean
    squared error between the data and the fit, the Pearson correlation, the
    Spearman correlation, and the latetr's p-value.

    Parameters
    ----------
    line : float
        Spectral line in Angstroms to be fit.
    wave : array
        Wavelengths in Angstroms.
    flux : array
        Flux.
    width : float, default=1.5
        The width of the local spectra centered about the line, i.e. the min
        and max wavelengths in Angstroms used when fitting is 
        [line - width, line + width]. A width of 1.5 usually contains around 
        40-90 data points.
    sigma_init : float, default=0.1
        The initial sigma guess in Angstroms.
    method : str, default='gauss'
        The fitting function used: single or double Gaussian/Lorentzian
        with/without a linear term.
    plot : bool, default=False
        If True, plot the data and the fit.

    Returns
    -------
    df_dict_line : dict
        The metadata, best fit parameters, and fit evaluation metrics.
    """

    # Reject values greater than 24000A (on unreliable part of spectrum)
    if line >= 24000:
        msg = f'Line needs to be under 24000 Angstroms; \
                currently measuring {line:.2f} Angstroms.'
        raise ValueError(msg)

    # Create dictionary and variables for plotting
    df_dict_line = {}
    line_str = str(round(line, 4))

    # Grab spectrum local centered at the OH line with a predetermined width
    mask_local = (wave >= line - width) & (wave <= line + width)
    wave_local = wave[mask_local]
    flux_local = flux[mask_local]

    # Mask out non finite values
    mask_finite = np.isfinite(flux_local)
    wave_local = wave_local[mask_finite]
    flux_local = flux_local[mask_finite]

    # Find and unpack inital parameters
    amp_init,  mu_init = find_initial_amp_mu(line, wave_local, flux_local)
    amp_init_1, amp_init_2 = amp_init
    mu_init_1, mu_init_2 = mu_init

    # Calculate and subtract median if not fitting linear function
    flux_local_median = np.nanmedian(flux_local)
    if 'linear' not in method:
        flux_local = flux_local - flux_local_median

    # Find parameter bounds and initial values
    bounds = find_parameter_bounds(line, flux_local, width=width)
    m_init = 0
    b_init = flux_local_median
    p0_init = np.array([amp_init_1, mu_init_1, sigma_init,
                        amp_init_2, mu_init_2, sigma_init,
                        m_init, b_init])

    # Choose function and initial values based on method
    func, p0 = initial_values_from_method(method, p0_init)

    # Fit line and predict flux
    try:
        popt, pcov = curve_fit(func, wave_local, flux_local,
                               p0=p0, maxfev=10000, bounds=bounds)
        condition = np.abs(popt[1] - line) > width
    # If all fits are outside of fitting bounds, optimal parameters are nan
        if len(popt) >= 6:
            condition_2 = np.abs(popt[4] - line) > width
            condition = condition & condition_2
        if condition:
            popt = np.ones(p0.shape[0]) * np.nan
            pcov = np.ones((p0.shape[0], p0.shape[0])) * np.nan
    # If model reaches iteration limit, optimal parameters are nan
    except RuntimeError:
        popt = np.ones(p0.shape[0]) * np.nan
        pcov = np.ones((p0.shape[0], p0.shape[0])) * np.nan
    flux_local_fit = func(wave_local, *popt)

    # Unpack best fit parameters
    if len(popt) == 3:
        amp_fit_1, mu_fit_1, sigma_fit_1 = popt
    elif len(popt) == 5:
        amp_fit_1, mu_fit_1, sigma_fit_1, m_fit, b_fit = popt
    elif len(popt) == 6:
        amp_fit_1, mu_fit_1, sigma_fit_1 = popt[:3]
        amp_fit_2, mu_fit_2, sigma_fit_2 = popt[3:]
    elif len(popt) == 8:
        amp_fit_1, mu_fit_1, sigma_fit_1 = popt[:3]
        amp_fit_2, mu_fit_2, sigma_fit_2 = popt[3:6]
        m_fit, b_fit = popt[6:]

    # Subtract continuum if fitting linear function
    continuum = 0
    if 'linear' in method:
        continuum += (m_fit * wave_local + b_fit)

    # Subtract "Gaussian/Lorentzian continuum" if fit is negligible
    if len(popt) >= 6:
        mu_ind_1 = np.argmin(np.abs(mu_fit_1 - wave_local))
        mu_ind_2 = np.argmin(np.abs(mu_fit_2 - wave_local))
        _, median_clipped, __ = sigma_clipped_stats(flux_local_fit, 
                                            sigma=3., 
                                            cenfunc='mean', 
                                            maxiters=None)
        offset = median_clipped * 1e-10
        diff_mu_1 = flux_local_fit[mu_ind_1] - median_clipped + offset
        diff_mu_2 = flux_local_fit[mu_ind_2] - median_clipped + offset
        peak_ratio = np.abs(diff_mu_1 / diff_mu_2)
        if peak_ratio > 10 or np.abs(mu_fit_2 - line) > width:
            continuum += func_gauss(wave_local, 
            amp_fit_2, mu_fit_2, sigma_fit_2)
        elif 1 / peak_ratio > 10 or np.abs(mu_fit_1 - line) > width:
            continuum += func_gauss(wave_local, 
            amp_fit_1, mu_fit_1, sigma_fit_1)

    flux_local_no_cont = flux_local - continuum
    flux_local_fit_no_cont = flux_local_fit - continuum

    # Integrate flux for observations and predictions using Simpson's rule
    flux_local_no_cont_I = simpson(flux_local_no_cont, wave_local)
    flux_local_fit_no_cont_I = simpson(flux_local_fit_no_cont, wave_local)

    # Calculate metrics
    flux_local_mad = median_abs_deviation(flux_local)
    error = flux_local_no_cont_I - flux_local_fit_no_cont_I
    rel_error = np.abs(error / flux_local_no_cont_I) * 100
    mse = np.mean((flux_local_no_cont - flux_local_fit_no_cont) ** 2)
    pearson = np.corrcoef(flux_local_no_cont, flux_local_fit_no_cont)[0,1]
    res = spearmanr(flux_local_no_cont, flux_local_fit_no_cont)
    spearman = res.correlation
    pvalue = res.pvalue

    # Record line metadata, fit parameters, and metrics
    df_dict_line['{}_median'.format(line_str)] = flux_local_median
    df_dict_line['{}_mad'.format(line_str)] = flux_local_mad
    df_dict_line['{}_sum'.format(line_str)] = flux_local_fit_no_cont_I
    df_dict_line['{}_amp1'.format(line_str)] = amp_fit_1
    df_dict_line['{}_mu1'.format(line_str)] = mu_fit_1
    df_dict_line['{}_sigma1'.format(line_str)] = np.abs(sigma_fit_1)
    df_dict_line['{}_error'.format(line_str)] = rel_error
    df_dict_line['{}_mse'.format(line_str)] = mse
    df_dict_line['{}_pearson'.format(line_str)] = pearson
    df_dict_line['{}_spearman'.format(line_str)] = spearman
    df_dict_line['{}_pvalue'.format(line_str)] = pvalue
    df_dict_line['{}_sigma1_cov'.format(line_str)] = pcov[2, 2]

    # Record more fit parameters based on method
    if len(popt) == 5:
        df_dict_line['{}_slope'.format(line_str)] = m_fit
        df_dict_line['{}_intercept'.format(line_str)] = b_fit
    if len(popt) == 6:
        df_dict_line['{}_amp2'.format(line_str)] = amp_fit_2
        df_dict_line['{}_mu2'.format(line_str)] = mu_fit_2
        df_dict_line['{}_sigma2'.format(line_str)] = np.abs(sigma_fit_2)
        df_dict_line['{}_ratio'.format(line_str)] = ratio
    if len(popt) == 8:
        df_dict_line['{}_amp2'.format(line_str)] = amp_fit_2
        df_dict_line['{}_mu2'.format(line_str)] = mu_fit_2
        df_dict_line['{}_sigma2'.format(line_str)] = np.abs(sigma_fit_2)
        df_dict_line['{}_sigma2_cov'.format(line_str)] = pcov[5, 5]
        df_dict_line['{}_slope'.format(line_str)] = m_fit
        df_dict_line['{}_intercept'.format(line_str)] = b_fit
        df_dict_line['{}_peak_ratio'.format(line_str)] = peak_ratio

    # Plot if needed
    flux_local_min = np.nanmin(flux_local)
    flux_local_max = np.nanmax(flux_local)
    r2 = pearson ** 2
    fs = 20
    title = f'{line_str} Line (fitting width:$\pm${width}) $R^2$:{r2:.3f}'
    if plot:
        plt.figure(figsize=[10,10])
        plt.title(title, fontsize=fs)
        # Plot flux_local
        plt.scatter(wave_local - line, flux_local,
                    label='data')
        # Plot flux_local_fit
        plt.plot(wave_local - line, flux_local_fit,
                 label='flux model fit', color='C1')
        # Plot mu_init_1
        label_mu1 = r'initial ${\mu}_{1}$'
        plt.vlines(mu_init_1 - line, flux_local_min, flux_local_max,
                   color='C2', alpha=0.5, linestyle='--', label=label_mu1)
        # Plot mu_init_2
        label_mu2 = r'initial ${\mu}_{2}$'
        plt.vlines(mu_init_2 - line, flux_local_min, flux_local_max,
                   color='C3', alpha=0.5, linestyle='--', label=label_mu2)

        plt.xlabel(f'Wavelength centered at {line_str} (A)', fontsize=fs)
        plt.ylabel('Relative Flux', fontsize=fs)
        plt.legend(fontsize=fs)
        plt.tick_params(labelsize=fs)
        plt.show()

    return df_dict_line

def fit_spectra(files, lines, normalize=True, width=1.5, method='gauss', 
                plot=False):
    """ Fit spectral lines and recorded metadata given a list of spectra.

    Parameters
    ----------
    files : list of str
        File names.
    lines : list of floats
        Spectral lines in Angstroms.
    normalize : bool, default=True
        If True, normalize flux by median.
    width : float, default=1.5
        The width of the local spectra centered about the line, i.e. the min
        and max wavelengths in Angstroms used when fitting is 
        [line - width, line + width]. A width of 1.5 usually contains around 
        40-90 data points.
    method : str, default='gauss'
        The fitting function used: single or double Gaussian/Lorentzian
        with/without a linear term.
    plot : bool, default=False
        If True, plot the data and the fit.

    Returns
    -------
    df : pd.DataFrame
        Table with all recorded metadata, best fit parameters, and evaluation
        metrics.
    """

    # Make dict
    df_dict = make_dict(lines, method)

    # Loading bar
    for i, file in tqdm(enumerate(files), total=len(files)):

        # Get wave, flux, exptime, and metadata
        wave, flux, df_dict_spec = get_wfe_metadata(file, normalize=normalize)
        for key in df_dict_spec.keys():
            df_dict[key].append(df_dict_spec[key])

        # Fit lines
        for line in lines:
            df_dict_line = fit_line(line, wave, flux, width=width,
                                    method=method, plot=plot)
            for key in df_dict_line.keys():
                df_dict[key].append(df_dict_line[key])

    # Convert dict to df and sort by MJD
    df = pd.DataFrame.from_dict(df_dict)

    return df

if __name__ == '__main__':
    
    # Unpack arguments
    data_dir = sys.argv[1]
    save_path = sys.argv[2]

    # Load lines and data
    lines_path = '../data/hydroxyl_lines_rousselot_2000.txt'
    lines_vac, lines_air = np.loadtxt(lines_path).T
    files = np.array(glob(f'{data_dir}/*'))

    # Fit lines and save to csv
    df = fit_spectra(files, lines_vac, normalize=True, 
                        method='gauss double linear', plot=False)
    df.to_csv(save_path)
