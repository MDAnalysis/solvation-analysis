
import numpy as np
import pytest

from solvation_analysis.pairing import Pairing


def test_pairing_from_solute(run_solute):
    pairing = Pairing.from_solute(run_solute)
    assert len(pairing.pairing_dict) == 3
    assert len(pairing.percent_free_solvents) == 3

@pytest.mark.parametrize(
    "name, percent",
    [
        ("fec", 0.21),
        ("bn", 1.0),
        ("pf6", 0.14),
    ],
)
def test_pairing_dict(name, percent, solvation_data):
    pairing = Pairing(solvation_data, 10, 49, {'fec': 237, 'bn': 363, 'pf6': 49})
    np.testing.assert_allclose(percent, pairing.pairing_dict[name], atol=0.05)
    assert len(pairing.pairing_by_frame) == 3


@pytest.mark.parametrize(
    "name, percent",
    [
        ("fec", 0.947),
        ("bn", 0.415),
        ("pf6", 0.853),
    ],
)
def test_pairing_participating(name, percent, solvation_data):
    pairing = Pairing(solvation_data, 10, 49, {'fec': 237, 'bn': 363, 'pf6': 49})
    np.testing.assert_allclose(percent, pairing.percent_free_solvents[name], atol=0.05)


@pytest.mark.parametrize(
    "name, diluent_percent",
    [
        ("fec", 0.54),
        ("bn", 0.36),
        ("pf6", 0.10),
    ],
)
def test_diluent_composition(name, diluent_percent, solvation_data):
    pairing = Pairing(solvation_data, 10, 49, {'fec': 237, 'bn': 363, 'pf6': 49})
    np.testing.assert_allclose(diluent_percent, pairing.diluent_dict[name], atol=0.05)
    np.testing.assert_allclose(sum(pairing.diluent_dict.values()), 1, atol=0.05)

