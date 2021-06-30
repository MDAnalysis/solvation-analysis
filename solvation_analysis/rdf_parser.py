import numpy as np
import numpy.polynomial.polynomial
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d, UnivariateSpline


def fit_data(bins, rdf):
    f = UnivariateSpline(bins, rdf, k=4, s=0)
    x = np.linspace(0, 10, num=100)
    y = f(x)
    return x, y


def identify_minima(bins, rdf):
    f = UnivariateSpline(bins, rdf, k=4, s=0)
    cr_pts = f.derivative().roots()
    cr_vals = f(cr_pts)
    return cr_pts, cr_vals