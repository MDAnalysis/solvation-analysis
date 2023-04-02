
import numpy as np
import pytest

from solvation_analysis.pairing import Pairing


def test_pairing_from_solute(run_solute):
    pairing = Pairing.from_solute(run_solute)
    assert len(pairing.solvent_pairing) == 3
    assert len(pairing.fraction_free_solvents) == 3


@pytest.mark.parametrize(
    "name, fraction",
    [
        ("fec", 0.21),
        ("bn", 1.0),
        ("pf6", 0.14),
    ],
)
def test_pairing_dict(name, fraction, solvation_data):
    pairing = Pairing(solvation_data, 10, 49, {'fec': 237, 'bn': 363, 'pf6': 49})
    np.testing.assert_allclose(fraction, pairing.solvent_pairing[name], atol=0.05)
    assert len(pairing.pairing_by_frame) == 3


@pytest.mark.parametrize(
    "name, fraction",
    [
        ("fec", 0.947),
        ("bn", 0.415),
        ("pf6", 0.853),
    ],
)
def test_pairing_participating(name, fraction, solvation_data):
    pairing = Pairing(solvation_data, 10, 49, {'fec': 237, 'bn': 363, 'pf6': 49})
    np.testing.assert_allclose(fraction, pairing.fraction_free_solvents[name], atol=0.05)


@pytest.mark.parametrize(
    "name, diluent_fraction",
    [
        ("fec", 0.54),
        ("bn", 0.36),
        ("pf6", 0.10),
    ],
)
def test_diluent_composition(name, diluent_fraction, solvation_data):
    pairing = Pairing(solvation_data, 10, 49, {'fec': 237, 'bn': 363, 'pf6': 49})
    np.testing.assert_allclose(diluent_fraction, pairing.diluent_composition[name], atol=0.05)
    np.testing.assert_allclose(sum(pairing.diluent_composition.values()), 1, atol=0.05)

