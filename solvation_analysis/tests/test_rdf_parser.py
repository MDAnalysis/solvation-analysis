import MDAnalysis as mda
import numpy as np

import matplotlib.pyplot as plt
import pytest
from solvation_analysis.rdf_parser import (
    identify_minima,
    interpolate_rdf,
    plot_interpolation_fit,
    identify_solvation_cutoff,
)


rdf_minima = [
    ("fec_F", []),
    ("fec_O", [2.9]),
    ("fec_all", [2.9]),
    ("bn_all", []),
    ("bn_N", []),
    ("pf6_all", []),
    ("pf6_F", []),
]


@pytest.mark.parametrize(
    "rdf_tag",
    ["fec_F", "fec_O", "fec_all", "bn_all", "bn_N", "pf6_all", "pf6_F"],
)
def test_plot_interpolation_fit(rdf_tag, rdf_bins_and_data_easy):
    """This is essentially a visually confirmed regression test to ensure
    behavior is approximately correct."""
    bins, rdf = rdf_bins_and_data_easy[rdf_tag]
    plot_interpolation_fit(bins, rdf)


@pytest.mark.parametrize(
    "rdf_tag, bounds",
    [
        ("fec_F", (0, 3)),
        ("fec_O", (0, 3)),
        ("fec_all", (0, 3)),
        ("bn_all", (0, 3)),
        ("bn_N", (0, 3)),
        ("pf6_all", (0, 3)),
        ("pf6_F", (0, 3)),
    ],  # the above values are not real
)
def test_interpolate_rdf(rdf_tag, bounds, rdf_bins_and_data_easy):
    bins, rdf = rdf_bins_and_data_easy[rdf_tag]
    return


@pytest.mark.parametrize(
    "rdf_tag, minima",
    [
        ("fec_F", [1, 2, 3]),
        ("fec_O", [1, 2, 3]),
        ("fec_all", [1, 2, 3]),
        ("bn_all", [1, 2, 3]),
        ("bn_N", [1, 2, 3]),
        ("pf6_all", [1, 2, 3]),
        ("pf6_F", [1, 2, 3]),
    ],  # the above values are not real
)
def test_identify_minima(rdf_tag, minima, rdf_bins_and_data_easy):
    bins, rdf = rdf_bins_and_data_easy[rdf_tag]
    return


@pytest.mark.parametrize(
    "rdf_tag, cutoff",
    [
        ("fec_F", 2.8),
        ("fec_O", 2.9),
        ("fec_all", 2.9),
        ("bn_all", 3),
        ("bn_N", 3),
        ("pf6_all", 3),
        ("pf6_F", 3),
    ],  # the above values are not real
)
def test_identify_solvation_cutoff(rdf_tag, cutoff, rdf_bins_and_data_easy):
    bins, rdf = rdf_bins_and_data_easy[rdf_tag]
    print(identify_solvation_cutoff(bins, rdf))
    return
