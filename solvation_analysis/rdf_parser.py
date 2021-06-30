import numpy as np
from scipy.interpolate import interp1d, UnivariateSpline
import scipy
import matplotlib.pyplot as plt


def interpolate_rdf(bins, rdf, floor=0.05, cutoff=5):
    start = np.argmax(rdf > floor)  # will return first index > rdf
    end = np.argmax(bins > cutoff)  # will return first index > cutoff
    bounds = (bins[start], bins[end - 1])
    f = UnivariateSpline(bins[start:end], rdf[start:end], k=4, s=0)
    return f, bounds


def identify_minima(f):
    assert isinstance(
        f, scipy.interpolate.fitpack2.InterpolatedUnivariateSpline
    ), "this function is compatible with the scipy.interpolate package"
    cr_pts = f.derivative().roots()
    cr_vals = f(cr_pts)
    return cr_pts, cr_vals


def plot_interpolation_fit(bins, rdf, floor=0.05, cutoff=5):
    f, bounds = interpolate_rdf(bins, rdf)
    x = np.linspace(bounds[0], bounds[1], num=100)
    y = f(x)
    pts, vals = identify_minima(f)
    plt.plot(bins, rdf, "b-")
    plt.plot(x, y, "r-")
    plt.plot(pts, vals, "go")
    plt.show()


def identify_solvation_cutoff(bins, rdf):
    return
