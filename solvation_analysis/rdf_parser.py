import numpy as np
from scipy.interpolate import interp1d, UnivariateSpline
import scipy
import matplotlib.pyplot as plt
import warnings


def interpolate_rdf(bins, rdf, floor=0.05, cutoff=5):
    start = np.argmax(rdf > floor)  # will return first index > rdf
    end = np.argmax(bins > cutoff)  # will return first index > cutoff
    bounds = (bins[start], bins[end - 1])
    f = UnivariateSpline(bins[start:end], rdf[start:end], k=4, s=0)
    return f, bounds


def identify_minima(f):
    try:
        cr_pts = f.derivative().roots()
        cr_vals = f(cr_pts)
        return cr_pts, cr_vals
    except AttributeError:
        print("f should be a scipy.interpolate.UnivariateSpline.")


def plot_interpolation_fit(bins, rdf, **kwargs):
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


def identify_solvation_cutoff(
    bins, rdf, failure_behavior="warn", cutoff_region=(1.5, 4), **kwargs
):
    f, bounds = interpolate_rdf(bins, rdf, **kwargs)
    cr_pts, cr_vals = identify_minima(f)
    if (
        len(cr_pts) < 2  # insufficient critical points
        or cr_vals[0] < cr_vals[1]  # not a max and min
        or not (cutoff_region[1] < cr_pts[1] < cutoff_region[4])  # min not in cutoff
    ):
        if failure_behavior == "silent":
            return None
        if failure_behavior == "warn":
            warnings.warn("No solvation region detected.")
            return None
        if failure_behavior == "exception":
            raise RuntimeError("No solvation region detected.")
    return cr_pts[1]
