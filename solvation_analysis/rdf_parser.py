import numpy as np
from scipy.interpolate import UnivariateSpline
import scipy
import matplotlib.pyplot as plt
import warnings


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
    Identifies whether or not the interpolation method has identified a valid
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
        or abs(cr_vals[1] - cr_vals[0]) < 0.15  # peak too small TODO: improve this!
    ):
        return False
    else:
        return True


def identify_solvation_cutoff(
    bins, rdf, failure_behavior="warn", cutoff_region=(1.5, 4), floor=0.05, cutoff=5
):
    """

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
            raise RuntimeError("Solution could not identify a solvation radius for at least one solvent. "
                               "Please enter the missing radii manually by adding them to the radii dict"
                               "and rerun the analysis.")
    return cr_pts[1]
