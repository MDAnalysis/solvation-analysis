import numpy as np
import pytest

from solvation_analysis.networking import Networking

from solvation_analysis._column_names import *


def test_networking_from_solute(run_solute):
    networking = Networking.from_solute(run_solute, 'pf6')
    assert len(networking.network_df) == 128


@pytest.mark.parametrize(
    "status, fraction",
    [
        (ISOLATED, 0.876),
        (PAIRED, 0.112),
        (NETWORKED, 0.012),
    ],
)
def test_get_cluster_res_ix(status, fraction, networking):
    np.testing.assert_almost_equal(networking.solute_status[status], fraction, 3)


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
