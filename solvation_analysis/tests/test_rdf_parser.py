import MDAnalysis as mda
import numpy as np

import matplotlib.pyplot as plt
import pytest
from solvation_analysis.rdf_parser import (
    identify_minima,
    interpolate_rdf,
    plot_interpolation_fit,
    plot_scipy_find_peaks_troughs,
    identify_cutoff_poly,
    identify_cutoff_scipy,
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
def test_plot_interpolation_fit(rdf_tag, rdf_bins_and_data_hard):
    """This is essentially a visually confirmed regression test to ensure
    behavior is approximately correct."""
    bins, rdf = rdf_bins_and_data_hard[rdf_tag]
    fig, ax = plot_interpolation_fit(bins, rdf)
    ax.set_title(f"Interpolation of RDF: {rdf_tag}")
    # plt.show()  # this should only be uncommented for local testing


@pytest.mark.parametrize(
    "rdf_tag", ["fec_F", "fec_O", "fec_all", "bn_all", "bn_N", "pf6_all", "pf6_F"]
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
    np.testing.assert_allclose(test_min, min, atol=0.01)


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
        ("fec_F", np.NaN),
        ("fec_O", 3.30),
        ("fec_all", 2.74),
        ("bn_all", 2.64),
        ("bn_N", 3.42),
        ("pf6_all", 2.77),
        ("pf6_F", 3.03),
    ],  # the above values are not real
)
def test_identify_cutoff_poly_easy(
    rdf_tag, cutoff, rdf_bins_and_data_easy, rdf_bins_and_data_hard
):
    bins, rdf = rdf_bins_and_data_easy[rdf_tag]
    np.testing.assert_allclose(
        identify_cutoff_poly(bins, rdf, failure_behavior="warn"),
        cutoff,
        atol=0.01,
        equal_nan=True,
    )

@pytest.mark.parametrize(
    "rdf_tag, cutoff",
    [
        ("fec_F", np.NaN),
        ("fec_O", 3.30),
        ("fec_all", 2.74),
        ("bn_all", 2.64),
        ("bn_N", 3.5),
        ("pf6_all", 2.77),
        ("pf6_F", 3.03),
    ],  # the above values are not real
)
def test_identify_cutoff_scipy_easy(
    rdf_tag, cutoff, rdf_bins_and_data_easy, rdf_bins_and_data_hard
):
    bins, rdf = rdf_bins_and_data_easy[rdf_tag]
    np.testing.assert_allclose(
        identify_cutoff_scipy(bins, rdf, failure_behavior="warn"),
        cutoff,
        atol=0.2,
        equal_nan=True,
    )

@pytest.mark.parametrize("rdf_tag", ["fec_F", "fec_all", "bn_all", "pf6_all", "pf6_F"])
def test_identify_cutoff_poly_hard(
    rdf_tag, rdf_bins_and_data_easy, rdf_bins_and_data_hard
):
    bins_ez, rdf_ez = rdf_bins_and_data_easy[rdf_tag]
    bins_hd, rdf_hd = rdf_bins_and_data_hard[rdf_tag]
    np.testing.assert_allclose(
        identify_cutoff_poly(bins_hd, rdf_hd, failure_behavior="warn"),
        identify_cutoff_poly(bins_ez, rdf_ez, failure_behavior="warn"),
        atol=0.1,
        equal_nan=True,
    )

@pytest.mark.parametrize("rdf_tag", ["fec_F", "fec_all", "bn_all", "pf6_all", "pf6_F"])
def test_identify_scipy_hard(
        rdf_tag, rdf_bins_and_data_easy, rdf_bins_and_data_hard
):
    bins_ez, rdf_ez = rdf_bins_and_data_easy[rdf_tag]
    bins_hd, rdf_hd = rdf_bins_and_data_hard[rdf_tag]
    np.testing.assert_allclose(
        identify_cutoff_scipy(bins_hd, rdf_hd, failure_behavior="warn"),
        identify_cutoff_scipy(bins_ez, rdf_ez, failure_behavior="warn"),
        atol=0.2,
        equal_nan=True,
    )


@pytest.mark.parametrize(
    "rdf_tag",
    ["fec_F", "fec_O", "fec_all", "bn_all", "bn_N", "pf6_all", "pf6_F"],
)
def test_plot_scripy_find_peaks_troughs(rdf_tag, rdf_bins_and_data_hard):
    """This is essentially a visually confirmed regression test to ensure
    behavior is approximately correct."""
    bins, rdf = rdf_bins_and_data_hard[rdf_tag]
    fig, ax = plot_scipy_find_peaks_troughs(bins, rdf)
    ax.set_title(f"Find peaks of RDF: {rdf_tag}")
    # plt.show()  # leave in, uncomment for debugging


def test_identify_cutoff_scipy_pf6(run_solute):
    pf6_bins, pf6_data = run_solute.rdf_data["solute_0"]["pf6"]
    radius = identify_cutoff_scipy(pf6_bins, pf6_data, failure_behavior="warn"),
    np.testing.assert_allclose(radius, 2.8, atol=0.2)


@pytest.mark.parametrize(
    "rdf_tag",
    [
        "fec_F_bn_N",
        "fec_O_bn_all",
        "fec_F_bn_all",
        "fec_F_pf6_all",
        "fec_F_pf6_F",
        "fec_O_pf6_F",
        "pf6_F_bn_N",
        "bn_N_fec_F",
        "bn_N_fec_all",
        "pf6_F_fec_all",
        "pf6_F_bn_all",
        "bn_N_pf6_F",
        "bn_N_pf6_all",
        "fec_O_bn_N",
        "fec_O_pf6_all",
        "pf6_F_fec_F",
        "bn_N_fec_O",
        "pf6_F_fec_O",
    ],
)
def test_identify_cutoff_non_solv(rdf_tag, rdf_bins_and_data_non_solv):
    bins, rdf = rdf_bins_and_data_non_solv[rdf_tag]
    np.testing.assert_allclose(
        identify_cutoff_poly(bins, rdf, failure_behavior="warn"),
        np.NaN,
        equal_nan=True,
    )
    np.testing.assert_allclose(
        identify_cutoff_scipy(bins, rdf, failure_behavior="warn"),
        np.NaN,
        equal_nan=True,
    )
