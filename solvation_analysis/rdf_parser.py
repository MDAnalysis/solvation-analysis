import numpy as np
from scipy.interpolate import interp1d, UnivariateSpline


def fit_data(bins, rdf, floor=0.05, cutoff=5):
    start = np.argmax(rdf > floor)  # will return first index > rdf
    end = np.argmax(bins > cutoff)  # will return first index > cutoff
    bounds = (bins[start], bins[end - 1])
    f = UnivariateSpline(bins[start:end], rdf[start:end], k=4, s=0)
    return f, bounds


def array_of_fit(bins, rdf):
    f, bounds = fit_data(bins, rdf)
    x = np.linspace(bounds[0], bounds[1], num=100)
    y = f(x)
    return x, y


def identify_minima(bins, rdf):
    f, bounds= fit_data(bins, rdf)
    cr_pts = f.derivative().roots()
    cr_vals = f(cr_pts)
    return cr_pts, cr_vals
