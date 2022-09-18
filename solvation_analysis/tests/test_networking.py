import numpy as np
import pytest

from solvation_analysis.networking import Networking


def test_networking_from_solution(run_solution):
    networking = Networking.from_solution(run_solution, 'pf6')
    assert len(networking.network_df) == 128


@pytest.fixture(scope='module')
def networking(run_solution):
    return Networking.from_solution(run_solution, 'pf6')


@pytest.mark.parametrize(
    "status, percent",
    [
        ('alone', 0.876),
        ('paired', 0.112),
        ('in_network', 0.012),
    ],
)
def test_get_cluster_res_ix(status, percent, networking):
    np.testing.assert_almost_equal(networking.solute_status[status], percent, 3)


@pytest.mark.parametrize(
    "network_ix, frame, n_res",
    [
        (0, 0, 3),
        (5, 1, 2),
        (1, 8, 3),
    ],
)
def test_get_network_res_ix(network_ix, frame, n_res, networking):
    res_ix = networking.get_network_res_ix(network_ix, frame)
    assert len(res_ix) == n_res
