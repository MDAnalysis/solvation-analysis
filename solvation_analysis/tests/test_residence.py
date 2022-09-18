import numpy as np
import pytest

from solvation_analysis.residence import Residence


def test_residence_from_solution(run_solution):
    residence = Residence.from_solution(run_solution)
    assert len(residence.residence_times) == 3
    assert len(residence.residence_times_fit) == 3


@pytest.fixture(scope='module')
def residence(solvation_data_sparse):
    return Residence(solvation_data_sparse, step=10)


@pytest.mark.parametrize(
    "name, res_time",
    [
        ("fec", 10),
        ("bn", 80),
        ("pf6", np.nan),
    ],
)
def test_residence_times(name, res_time, residence):
    np.testing.assert_almost_equal(residence.residence_times[name], res_time, 3)


@pytest.mark.parametrize("name", ['fec', 'bn', 'pf6'])
def test_plot_auto_covariance(name, residence):
    residence.plot_auto_covariance(name)


def test_residence_time_warning(solvation_data_sparse):
    # we step through the data frame to speed up the tests
    with pytest.warns(UserWarning, match="the autocovariance for pf6 does not converge"):
        Residence(solvation_data_sparse, step=10)
