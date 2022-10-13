"""
================
RDF Parser
================
:Author: Orion Cohen, Hugo MacDermott-Opeskin
:Year: 2021
:Copyright: GNU Public License v3

RDF Parser defines several functions for finding the solvation cutoff
from an RDF.
"""

import numpy as np
from scipy.interpolate import UnivariateSpline
import scipy
import matplotlib.pyplot as plt
import warnings
from scipy.signal import find_peaks, gaussian

from solvation_analysis._column_names import *


def interpolate_rdf(bins, rdf, floor=0.05, cutoff=5):
    """
    Fits a sciply.interpolate.UnivariateSpline to the starting region of
    the RDF. The floor and cutoff control the region of the RDF that the
    spline is fit to.

    Parameters
    ----------
    bins : np.array
        the x-axis bins of the rdf
    rdf : np.array
        rdf data matching the bins
    floor : float
        the interpolation region begins when the probability density value exceeds
        the floor
    cutoff : float
        the interpolation region ends when the bins values exceeds the cutoff
    Returns
    -------
    f, bounds : the interpolated spline and the bounds of the interpolation region

    """
    start = np.argmax(rdf > floor)  # will return first index > rdf
    end = np.argmax(bins > cutoff)  # will return first index > cutoff
    bounds = (bins[start], bins[end - 1])
    f = UnivariateSpline(bins[start:end], rdf[start:end], k=4, s=0)
    return f, bounds


def identify_minima(f):
    """
    Identifies the extrema of a interpolated polynomial.

    Parameters
    ----------
    f : sciply.interpolate.UnivariateSpline

    Returns
    -------

    cr_pts, cr_vals : the critical points and critical values of the extrema

    """
    try:
        isinstance(f, scipy.interpolate.UnivariateSpline)
    except AttributeError:
        print(
            "identify minima is designed to work with a",
            " scipy.interpolate.UnivariateSpline output by interpolate_rdf.",
        )
    cr_pts = f.derivative().roots()
    cr_vals = f(cr_pts)
    return cr_pts, cr_vals


def plot_interpolation_fit(bins, rdf, **kwargs):
    """
    Calls interpolate_rdf and identify_minima to identify the extrema of an RDF.
    Plots the original rdf, the interpolated spline, and the extrema of the
    interpolated spline.

    Parameters
    ----------
    bins : np.array
        the x-axis bins of the rdf
    rdf : np.array
        RDF data matching the bins
    kwargs : passed to the interpolate_rdf function

    Returns
    -------
    fig, ax : matplotlib pyplot Figure and Axis for the fit

    """
    f, bounds = interpolate_rdf(bins, rdf, **kwargs)
    x = np.linspace(bounds[0], bounds[1], num=100)
    y = f(x)
    pts, vals = identify_minima(f)
    fig, ax = plt.subplots()
    ax.plot(bins, rdf, "b--", label="rdf")
    ax.plot(x, y, "r-", label="interpolation")
    ax.plot(pts, vals, "go", label="critical points")
    ax.set_xlabel("Radial Distance (A)")
    ax.set_ylabel("Probability Density")
    ax.set_title("Interpolation of RDF with quartic spline")
    ax.legend()
    return fig, ax


def good_cutoff(cutoff_region, cr_pts, cr_vals):
    """
    Uses several heuristics to determine if the a solvation cutoff is valid
    solvation cutoff. This fails if there is no solvation shell.

    Parameters
    ----------
    f : an interpolated RDF function
    cutoff_region : tuple
        boundaries in which to search for a solvation shell cutoff, i.e. (1.5, 4)
    cr_pts : np.array
        the x-axis values of the extrema
    cr_vals : np.array
        the y-axis values of the extrema
    Returns
    -------
    boolean : True if good cutoff, False if bad cutoff

    """
    if (
        len(cr_pts) < 2  # insufficient critical points
        or cr_vals[0] < cr_vals[1]  # not a min and max
        or not (cutoff_region[0] < cr_pts[1] < cutoff_region[1])  # min not in cutoff
        or abs(cr_vals[1] - cr_vals[0]) < 0.15  # peak too small
    ):
        return False
    else:
        return True


def good_cutoff_scipy(cutoff_region, min_trough_depth, peaks, troughs, rdf, bins):
    """
    Uses several heuristics to determine if the solvation cutoff is valid
    solvation cutoff. This fails if there is no solvation shell.

    Heuristics:
      -  trough follows peak
      -  in `Solute.cutoff_region` (specified by kwarg)
      -  normalized peak height > 0.05

    Parameters
    ----------
    cutoff_region : tuple
        boundaries in which to search for a solvation shell cutoff, i.e. (1.5, 4)
    min_trough_depth : float
        the minimum depth of a trough to be considered a valid solvation cutoff
    peaks : np.array
        the indices of the peaks in the bins array
    troughs : np.array
        the indices of the troughs in the bins array
    bins : np.array
        the x-axis bins of the rdf
    rdf : np.array
        RDF data matching the bins
    Returns
    -------
    boolean : True if good cutoff, False if bad cutoff

    """
    # normalize rdf
    norm_rdf = rdf / np.max(rdf)
    if (
        len(peaks) == 0 or len(troughs) == 0  # insufficient critical points
        or troughs[0] < peaks[0]  # not a min and max
        or not (cutoff_region[0] < bins[troughs[0]] < cutoff_region[1])  # min not in cutoff
        or abs(norm_rdf[peaks[0]] - norm_rdf[troughs[0]]) < min_trough_depth  # peak too small
    ):
        return False
    return True


def scipy_find_peaks_troughs(bins, rdf, return_rdf=False, **kwargs):
    """
    Finds the indices of the peaks and troughs of an RDF.

    This function applies the following procedure to identify peaks.
        1. normalize the RDF
        2. apply a gaussian convolution (std=1.1) to the RDF
        3. call scipy.signal.find_peaks on the RDF and -1*RDF to find the peaks and troughs, respectively.

    Parameters
    ----------
    bins : np.array
        the x-axis bins of the rdf
    rdf : np.array
        RDF data matching the bins
    return_rdf : bool
        if True, returns the smoothed rdf after the peaks and troughs
    kwargs : dict
        passed to scipy.signal.find_peaks

    Returns
    -------
    default
        peaks (np.array), troughs (np.array)
    if return_rdf
        peaks (np.array), troughs (np.array), smoothed_rdf (np.array)

    """
    # normalize rdf
    norm_rdf = rdf / np.max(rdf)
    # this will set window_width to an odd number of bins that spans ~ 1.5 A
    bin_distance = bins[1] - bins[0]
    window_width = np.ceil(1.5 / bin_distance) // 2 * 2 + 1
    convolution = gaussian(window_width, 1.1)
    smooth_rdf = np.convolve(norm_rdf, convolution / np.sum(convolution), mode="same")
    troughs, _ = find_peaks(-smooth_rdf, **kwargs)
    peaks, _ = find_peaks(smooth_rdf)
    if return_rdf:
        return peaks, troughs, smooth_rdf * np.max(rdf)
    return peaks, troughs


def identify_cutoff_scipy(
    bins,
    rdf,
    cutoff_region=(1.5, 4),
    failure_behavior="warn",
    min_trough_depth=0.02,
    **kwargs
):
    """
    Identifies the solvation cutoff of an RDF.

    This function is a thin wrapper on scipy_find_peaks_trough, see the documentation
    of that function for more detail. It applies a few simple heuristics, specified in
    good_cutoff_scipy, to determine if the solvation cutoff is valid.

    Parameters
    ----------
    bins : np.array
        the x-axis bins of the rdf
    rdf : np.array
        RDF data matching the bins
    cutoff_region : tuple
        boundaries in which to search for a solvation shell cutoff, i.e. (1.5, 4)
    failure_behavior : str
        specifies the behavior of the function if no solvation shell is found, can
        be set to "silent", "warn", or "exception"
    min_trough_depth : float
        the minimum depth of a trough to be considered a valid solvation cutoff
    kwargs : passed to the scipy.find_peaks function

    Returns
    -------
    cutoff : float
        the solvation cutoff of the RDF
    """
    peaks, troughs = scipy_find_peaks_troughs(bins, rdf, **kwargs)
    if not good_cutoff_scipy(cutoff_region, min_trough_depth, peaks, troughs, rdf, bins):
        if failure_behavior == "silent":
            return np.NaN
        if failure_behavior == "warn":
            warnings.warn("No solvation shell detected.")
            return np.NaN
        if failure_behavior == "exception":
            raise RuntimeError("Solute could not identify a solvation radius for at least one solvent. "
                               "Please enter the missing radii manually by adding them to the radii dict"
                               "and rerun the analysis.")
    cutoff = bins[troughs[0]]
    return cutoff


def plot_scipy_find_peaks_troughs(
    bins,
    rdf,
    **kwargs,
):
    """
    Plot the original and smoothed RDF with the peaks and troughs located.

    This function is a thin wrapper on scipy_find_peaks_trough, see the documentation
    of that function for more detail.

    Parameters
    ----------
    bins : np.array
        the x-axis bins of the rdf
    rdf : np.array
        RDF data matching the bins
    kwargs : dict
        passed to scipy.signal.find_peaks

    Returns
    -------
    fig, ax : matplotlib pyplot Figure and Axis for the fit

    """
    peaks, troughs, smooth_rdf = scipy_find_peaks_troughs(bins, rdf, return_rdf=True, **kwargs)
    fig, ax = plt.subplots()
    ax.plot(bins, rdf, "b--", label="rdf")
    ax.plot(bins, smooth_rdf, "g-", label="smooth_rdf")
    ax.plot(bins[troughs], rdf[troughs], "go", label="troughs")
    ax.plot(bins[peaks], rdf[peaks], "ro", label="peaks")
    ax.set_xlabel("Radial Distance (A)")
    ax.set_ylabel("Probability Density")
    ax.set_title("RDF minima using scipy.find_peaks")
    ax.legend()
    return fig, ax


def identify_cutoff_poly(
    bins, rdf, failure_behavior="warn", cutoff_region=(1.5, 4), floor=0.05, cutoff=5
):
    """
    Identifies the solvation cutoff of an RDF using a polynomial interpolation.

    Parameters
    ----------
    bins : np.array
        the x-axis bins of the rdf
    rdf : np.array
        RDF data matching the bins
    failure_behavior : str
        specifies the behavior of the function if no solvation shell is found, can
        be set to "silent", "warn", or "exception"
    cutoff_region : tuple
        boundaries in which to search for a solvation shell cutoff, i.e. (1.5, 4)
    cutoff : float
        passed to the interpolate_rdf function
    floor : float
        passed to the interpolate_rdf function

    Returns
    -------
    float : the first solvation cutoff

    """
    f, bounds = interpolate_rdf(bins, rdf, floor=floor, cutoff=cutoff)
    cr_pts, cr_vals = identify_minima(f)
    if not good_cutoff(cutoff_region, cr_pts, cr_vals):
        if failure_behavior == "silent":
            return np.NaN
        if failure_behavior == "warn":
            warnings.warn("No solvation shell detected.")
            return np.NaN
        if failure_behavior == "exception":
            raise RuntimeError("Solute could not identify a solvation radius for at least one solvent. "
                               "Please enter the missing radii manually by adding them to the radii dict"
                               "and rerun the analysis.")
    return cr_pts[1]
