import numpy as np
import pytest

from solvation_analysis.coordination import Coordination


def test_coordination_from_solute(run_solute):
    coordination = Coordination.from_solute(run_solute)
    assert len(coordination.cn_dict) == 3


@pytest.mark.parametrize(
    "name, cn",
    [
        ("fec", 0.25),
        ("bn", 4.33),
        ("pf6", 0.15),
    ],
)
def test_coordination(name, cn, solvation_data, run_solute):
    atoms = run_solute.u.atoms
    coordination = Coordination(solvation_data, 10, 49, atoms)
    np.testing.assert_allclose(cn, coordination.cn_dict[name], atol=0.05)
    assert len(coordination.cn_by_frame) == 3


@pytest.mark.parametrize(
    "name, atom_type, fraction",
    [
        ("fec", '19', 0.008),
        ("bn", '5', 0.9976),
        ("pf6", '21', 1.000),
    ],
)
def test_coordinating_atoms(name, atom_type, fraction, solvation_data, run_solute):
    atoms = run_solute.u.atoms
    coordination = Coordination(solvation_data, 10, 49, atoms)
    calculated_fraction = coordination.coordinating_atoms.loc[(name, atom_type)]
    np.testing.assert_allclose(fraction, calculated_fraction, atol=0.05)














