""" Fit periodograms to hydroxyl fluxes measured using SPIRou sky spectra.

This script reads reads the saved .csv file from fit_lines.py, fits periodograms
for Events 1 and 2, and saves the most dominant periods (and their powers)
along with fitting metrics such as relative error, mean squared error, Pearson 
correlation, and Spearman correlation. It defaults to saving the top 5 periods
and disabling plotting. Changing these parameters can save any number of periods
and enable plotting.

Notes
-----
- The script finds the best frequencies by convolving on the periodogram since
the abundance of frequency bins cause redundancies.
- The time series are reconstructed by adding single frequency models and
recentering using a mean offset from the original time series.
- The minimum and maximum periods searched are 4 minutes (approx. period of 
observation) and 1 year.
- Event 1 has some observations with stars passing, which results in a spike in
flux so those observations are masked out.

Use
---
    This script can be run from the command line:

        >>> python fit_periodograms.py [save_path1] [save_path2] [file_csv]

    The user must provide three arguments: save_path1 and save_path2, which 
    corresponds to the path to save the output files for Events 1 and 2,
    respectively. The third is file_csv, which is the path to the output file
    from fit_lines.py

    This script can also be imported in a jupyter notebook:

        >>> from fit_periodograms import fit_all_periodograms

"""

import sys
from tqdm import tqdm

import numpy as np
from matplotlib import pyplot as plt
import pandas as pd
from scipy.stats import sigmaclip, spearmanr
from scipy.signal import argrelextrema

from astropy.timeseries import LombScargle
from astropy.convolution import convolve, Gaussian1DKernel

def find_best_freq(frequency, power, line, time, n=5, plot=False, xlim=[]):
    """ Find the n best frequencies that explain the data by convolving the 
    periodogram so redundant frequencies (e.g. within +/-2) don't all get 
    included. 

    Parameters
    ----------
    frequency : array
        The frequency bins from the Lomb-Scargle Periodogram.
    power : array
        The relative power of the frequency bins.
    line : str
        The spectral line being measured.
    time : float
        The start MJD of the data.
    n : float, default=5
        The number of frequencies to keep.
    plot : bool, default=True
        If True, plot the periodogram with the best frequencies.
    xlim : list, default=[]
        The minimum and maximum plotting values.

    Returns
    -------
    best_fs : array
        The best n frequencies.
    best_ps : array
        The relative powers corresponding to the best frequencies.
    """

    # Find nth best frequencies in conv space
    kernel = Gaussian1DKernel(stddev=16)
    power_conv = convolve(power, kernel)
    arx = argrelextrema(power_conv, np.greater)[0]
    best_p_cand = np.sort(power_conv[arx])[::-1][:n]
    mask_power = np.isin(power_conv, best_p_cand)
    best_fs = frequency[mask_power]
    best_ps = power[mask_power]
    
    # Find the best frequencies in real space
    best_fs_real = []
    for f in best_fs:
        mask_f = (frequency > f - 2) & (frequency < f + 2)
        ind = np.argmax(power[mask_f])
        best_fs_real.append(frequency[mask_f][ind])
    best_fs = np.array(best_fs_real)
    best_ps = power[np.isin(frequency, best_fs)]
    
    # Plot
    if plot:
        plt.figure(figsize=[30,10])
        unit = '$\mathring{A}$'
        sub_title = f'{line}{unit}, time={time:.2f}'
        plt.title(f'Lomb-Scargle Periodogram: {sub_title}', fontsize=30)
        plt.scatter(frequency, power) 
        plt.plot(frequency, power, label='power')
        plt.scatter(best_fs, best_ps, color='C2', s=100, label='local max')
        plt.hlines(np.nanmin(best_ps), frequency.min(), frequency.max(), 
                   color='C3', label='cutoff power')
        if xlim:
            plt.xlim(xlim[0], xlim[1])
        plt.xlabel('Frequency (per day)', fontsize=30)
        plt.ylabel('Power', fontsize=30)
        plt.legend(fontsize=30)
        plt.tick_params(labelsize=30)
        plt.show()
    
    return best_fs, best_ps

def reconstruct_lsp_fit(t, y, best_fs, forecast_size=10000):
    """Reconstruct time series using frequencies from Lomb-Scargle Periodogram.

    We reconstruct the time series by adding simple singular frequency models
    using the best frequencies. Since these simple models all have their own
    offset, we recenter the summation by finding the mean difference between
    the data and model, and subtract that difference from the model. This
    method has be shown emperically to well align the data.

    Parameters
    ----------
    t : array
        The time axis.
    y : array
        The signal of the time series.
    best_fs : array
        The best n frequencies.
    forecast_size : float, default=10000
        The number of data points for a high time resolution fit between the
        minimum and maximum time values.

    Returns
    -------
    y_fit : array
        The uncentered reconstruction at times from t.
    t_forecast : array
        A time array ranging between the minimum and maximum t values
        containing forecast_size number of data points. This acts as a high
        time resolution.
    y_forecast : array
        The uncentered reconstruction at times from t_forecast.
    offset_mean : float
        The mean difference between the uncentered reconstruction and the
        observed signal. Subtracting this from y_fit and y_forecast centers
        the reconstructions.
    """

    # Reconstruct time series
    ls = LombScargle(t, y)
    y_fit = np.zeros(t.shape)
    t_forecast = np.linspace(t.min(),t.max(),forecast_size)
    y_forecast = np.zeros(t_forecast.shape)
    for freq in best_fs:
        y_fit_f = ls.model(t, freq)
        y_fit += y_fit_f
        y_forecast_f = ls.model(t_forecast, freq)
        y_forecast += y_forecast_f

    # Find mean offset
    offset_pred = y_fit - y
    offset_mean = offset_pred.mean()

    return y_fit, t_forecast, y_forecast, offset_mean

def fit_one_periodogram(line, mask_event, time_centered, df, n=5, 
                        order_power=True, plot=False):
    """ Fit one line time series to a Lomb-Scargle Periodogram.

    This function performs the following steps:
    1. Mask out any fluxes:
        a. outside a specified time domain,
        b. that are NaN
        c. with > 15% relative error.
    2. Calculate the Lomb-Scargle Periodogram with automatic frequency bins
    and the fastest frequency being one cycle per second.
    3. Find the n best frequencies for reconstructions.
    4. Order best frequencies by decreasing relative power.
    5. Reconstruct time series using the best frequencies.
    6. Calculate mean squared error and coefficient of determination.

    Parameters
    ----------
    line : str
        The spectral line.
    mask_event : array
        A masking array to only fit a smaller set of data.
    time_centered : array
        The time array with 0 being the first observation.
    df : pandas.DataFrame
        The dataframe containing metadata, fit parameters, and metrics for
        curve fitting various spectral lines.
    n : float, default=5
        The number of frequencies to keep from the periodogram.
    order_power : bool, default=True
        If True, order the frequencies by decreasing relative power.
    plot : bool, default=False
        If True, plot the reconstructions.
    
    Returns
    -------
    row_entry : array
        The best frequencies in decreasing power order with mean squared error
        and coefficient of determination for each spectral line time series.
    """
    
    # Define flux and error
    line_flux = df[f'{line}_sum']
    line_error = df[f'{line}_error']

    # Mask out bad fluxes
    mask_nan = ~np.isnan(line_flux)
    mask_error = line_error < 15
    mask = mask_event & mask_nan & mask_error
    t = time_centered[mask]
    y = line_flux[mask]

    # LS Periodogram
    # 1 cycle / year * 1 year / 365 days = 1/365 cycles / day
    min_freq = 1 / 365
    # 1 cycle / 4 min * 60 min / 1 hr * 24 hr / 1 day = 15 * 24 cycles / day
    max_freq = 15 * 24
    frequency, power = LombScargle(t, y).autopower(
        minimum_frequency=min_freq, maximum_frequency=max_freq
        )

    # Find best frequencies
    best_fs, best_ps = find_best_freq(frequency, power, line, np.nanmin(t), n, 
                                      plot=plot)

    # Order from highest power to lowest by power
    if order_power:
        inds_sort = np.flip(np.argsort(best_ps))
        best_fs = best_fs[inds_sort]
        best_ps = best_ps[inds_sort]

    # Fit LS to best frequencies and calculate offset
    y_fit, t_forecast, y_forecast, offset_mean = reconstruct_lsp_fit(t, y, 
                                                                     best_fs)
    y_corrected = y_fit - offset_mean
    y_forecast_corrected = y_forecast - offset_mean

    #Plot
    unit = '$\mathring{A}$'
    title = f'Total Flux Contribution of {line}{unit}'
    if plot:
        plt.figure(figsize=[30,10])
        plt.title(title, fontsize=30)
        plt.plot(t, y, label='data')
        plt.scatter(t, y)
        plt.plot(t_forecast, y_forecast_corrected, label='LSP reconstruction', alpha=0.5)
        plt.xlabel('Days from First Observation ($MJD_0$=58327)', fontsize=30)
        plt.ylabel('Normalized Flux', fontsize=30)
        plt.legend(fontsize=30)
        plt.tick_params(labelsize=30)
        plt.show()

    # Calculate metrics
    error = np.median(np.abs((y_corrected - y) / y)) * 100
    mse = np.mean((y_corrected - y)**2)
    pearson = np.corrcoef(y_corrected, y)[0, 1]
    spearman = spearmanr(y_corrected, y).correlation
    metrics = np.array([error, mse, pearson, spearman])

    # Create one row entry
    row_entry = np.concatenate((best_fs, best_ps, metrics))

    return row_entry

def fit_all_periodograms(file_csv, n=5, plot=False):
    """ Fit Lomb-Scargle Periodograms to all spectral lines.

    Parameters
    ----------
    file_csv : str:
        Path to the csv file with the line metadata, fit parameters, and 
        metrics.
    n : float, default=5
        The number of frequencies to keep from the periodogram.
    plot : bool, default=False
        If True, plot the periodogram with the best frequencies.

    Returns
    -------
    df_mask1 : pandas.DataFrame
        The dataframe with the best frequencies, corresponding powers, and 
        fitting metrics for Event 1 (first three day period).
    df_mask2 : pandas.DataFrame
        The dataframe with the best frequencies, corresponding powers, and 
        fitting metrics for Event 2 (second three day period).
    """
    
    # Define lines and time array
    df = pd.read_csv(file_csv, index_col=0).sort_values(by='spec_mjd')
    lines_path = '../data/hydroxyl_lines_rousselot_2000.txt'
    lines_vac, lines_air = np.loadtxt(lines_path).T
    lines = lines_vac.astype(str)
    time_centered = df['spec_mjd'].values - int(df['spec_mjd'].min())
    
    # Define masks without star passing (3 day periods of night observations)
    mask1_1 = (time_centered > 504) & (time_centered < 504.5)
    mask1_2 = (time_centered > 504.58) & (time_centered < 505.515)
    mask1_3 = (time_centered > 505.58) & (time_centered < 506.578)
    mask1_4 = (time_centered > 506.61) & (time_centered < 507)
    mask1 = mask1_1 | mask1_2 | mask1_3 | mask1_4 #32%
    mask2 = (time_centered > 543) & (time_centered < 547) #35%

    # Fit LS for all lines on given timescales
    table = []
    for mask_event in [mask1, mask2]:
        table_mask = []
        for line in tqdm(lines, total=lines.shape[0]):
            row_entry = fit_one_periodogram(line, mask_event, time_centered, df, 
                                            n=n, plot=plot)
            table_mask.append(row_entry)
        table.append(table_mask)
    
    # Make DataFrame
    table_mask1, table_mask2 = table

    columns_f = []
    columns_p = []
    for i in range (n):
        columns_f.append(f'f{i+1}')
        columns_p.append(f'p{i+1}')
    columns = columns_f + columns_p
    columns = columns + ['error', 'mse', 'pearson', 'spearman']

    df_mask1 = pd.DataFrame(table_mask1, index=lines, columns=columns)
    df_mask2 = pd.DataFrame(table_mask2, index=lines, columns=columns)

    return df_mask1, df_mask2

if __name__ == '__main__':

    # Unpack arguments
    save_path1 = sys.argv[1]
    save_path2 = sys.argv[2]
    file_csv = sys.argv[3]

    # Fit periodograms and save as csv files
    df1, df2 = fit_all_periodograms(file_csv, n=5, plot=False)
    df1.to_csv(save_path1)
    df1.to_csv(save_path2)

