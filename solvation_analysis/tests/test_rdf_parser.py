import MDAnalysis as mda
import numpy as np

import matplotlib.pyplot as plt
import pytest
from solvation_analysis.rdf_parser import (
    identify_minima,
    interpolate_rdf,
    plot_interpolation_fit
)

rdf_parameters = [
    ("fec_F", []),
    ("fec_O", [2.9]),
    ("fec_all", [2.9]),
    ("bn_all", []),
    ("bn_N", []),
    ("pf6_all", []),
    ("pf6_F", []),
]


@pytest.mark.parametrize("rdf_tag, minima", rdf_parameters)
def test_plot_interpolation_fit(rdf_tag, minima, rdf_bins, rdf_data):
    """This is essentially a visually confirmed regression test to ensure
    behavior is approximately correct."""
    bins = rdf_bins[rdf_tag]
    rdf = rdf_data[rdf_tag]
    plot_interpolation_fit(bins, rdf)


@pytest.mark.parametrize("rdf_tag, minima", rdf_parameters)
def test_interpolate_rdf(rdf_tag, minima, rdf_bins, rdf_data):
    return
