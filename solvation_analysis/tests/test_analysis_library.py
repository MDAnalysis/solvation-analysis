import numpy as np
import pytest

from solvation_analysis.analysis_library import (
    _Speciation,
    _Coordination,
    _Pairing,
    _SolvationData
)


@pytest.mark.parametrize(
    "cluster, percent",
    [
        ({'bn': 5, 'fec': 0, 'pf6': 0}, 0.357),
        ({'bn': 3, 'fec': 3, 'pf6': 0}, 0.004),
        ({'bn': 3, 'fec': 0, 'pf6': 1}, 0.016),
    ],
)
def test_ion_speciation(cluster, percent, solvation_data):
    speciation = _Speciation(solvation_data, 10, 49)
    speciation.find_clusters({'bn': 5})
    percentage = speciation.cluster_percent(cluster)
    np.testing.assert_allclose(percent, percentage, atol=0.05)


@pytest.mark.parametrize(
    "name, cn",
    [
        ("fec", 0.25),
        ("bn", 4.33),
        ("pf6", 0.15),
    ],
)
def test_coordination_numbers(name, cn, solvation_data):
    coord_dict = _Coordination(solvation_data, 10, 49).cn_dict
    np.testing.assert_allclose(cn, coord_dict[name], atol=0.05)


@pytest.mark.parametrize(
    "name, percent",
    [
        ("fec", 0.21),
        ("bn", 1.0),
        ("pf6", 0.14),
    ],
)
def test_pairing(name, percent, solvation_data):
    pairing_dict = _Pairing(solvation_data, 10, 49).percentage_dict
    np.testing.assert_allclose([percent], pairing_dict[name], atol=0.05)
