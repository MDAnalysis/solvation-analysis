import MDAnalysis as mda
import numpy as np

import matplotlib.pyplot as plt
import pytest
from solvation_analysis.rdf_parser import (
    identify_minima,
    interpolate_rdf,
    plot_interpolation_fit,
    identify_solvation_cutoff,
    good_cutoff,
)
from scipy.interpolate import UnivariateSpline
import scipy

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
    fig, ax = plot_interpolation_fit(bins, rdf)
    ax.set_title(f"Interpolation of RDF: {rdf_tag}")
    plt.show()


@pytest.mark.parametrize(
    "rdf_tag",
    ["fec_F", "fec_O", "fec_all", "bn_all", "bn_N", "pf6_all", "pf6_F"]
)
def test_interpolate_rdf(rdf_tag, rdf_bins_and_data_easy):
    # this is difficult to test so only very basic checks are implemented
    bins, rdf = rdf_bins_and_data_easy[rdf_tag]
    f, bounds = interpolate_rdf(bins, rdf)
    assert bounds[0] > 0
    assert bounds[1] <= 5
    assert isinstance(f, scipy.interpolate.fitpack2.InterpolatedUnivariateSpline)


@pytest.mark.parametrize(
    "rdf_tag, test_min",
    [
        ("fec_O", 3.30),
        ("fec_all", 2.74),
        ("bn_all", 2.64),
        ("pf6_all", 2.77),
        ("pf6_F", 3.03),
    ],  # the above values are not real
)
def test_identify_minima_first_min(rdf_tag, test_min, rdf_bins_and_data_easy):
    bins, rdf = rdf_bins_and_data_easy[rdf_tag]
    f, bounds = interpolate_rdf(bins, rdf)
    cr_pts, cr_vals = identify_minima(f)
    min = cr_pts[1]
    np.testing.assert_almost_equal(test_min, min, 2)


# def test_identify_minima_second_min(rdf_tag, test_min, rdf_bins_and_data_easy):
#     # later on this should test the identification of a second minimum
#     return


@pytest.mark.parametrize(
    "cutoff_region, cr_pts, cr_vals, expected",
    [
        ((1, 4), [2], [2], False),
        ((1, 4), [2, 3], [1.0, 1.1], False),
        ((1, 1.5), [2, 3], [2, 1], False),
        ((1.5, 4), [2, 3], [2, 1], True),
    ],
)
def test_good_cutoff(cutoff_region, cr_pts, cr_vals, expected):
    assert good_cutoff(cutoff_region, cr_pts, cr_vals) == expected


@pytest.mark.parametrize(
    "rdf_tag, cutoff",
    [
        ("fec_F", 2.73),
        ("fec_O", 2.9),
        ("fec_all", 2.74),
        ("bn_all", 2.64),
        ("bn_N", 3),
        ("pf6_all", 2.77),
        ("pf6_F", 3),
    ],  # the above values are not real
)
def test_identify_solvation_cutoff(rdf_tag, cutoff, rdf_bins_and_data_easy):
    bins, rdf = rdf_bins_and_data_easy[rdf_tag]
    cutoff = identify_solvation_cutoff(bins, rdf, failure_behavior='warn')
    print(cutoff)
