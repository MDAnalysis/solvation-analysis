import numpy as np
import pytest

from solvation_analysis.analysis_library import (
    Speciation,
    Coordination,
    Pairing,
)


@pytest.mark.parametrize(
    "cluster, percent",
    [
        ({'bn': 5, 'fec': 0, 'pf6': 0}, 0.357),
        ({'bn': 3, 'fec': 3, 'pf6': 0}, 0.004),
        ({'bn': 3, 'fec': 0, 'pf6': 1}, 0.016),
        ({'bn': 4}, 0.531),
    ],
)
def test_speciation_cluster_percent(cluster, percent, solvation_data):
    speciation = Speciation(solvation_data, 10, 49)
    percentage = speciation.shell_percent(cluster)
    np.testing.assert_allclose(percent, percentage, atol=0.05)


@pytest.mark.parametrize(
    "cluster, n_clusters",
    [
        ({'bn': 5, 'fec': 0, 'pf6': 0}, 175),
        ({'bn': 3, 'fec': 3, 'pf6': 0}, 2),
        ({'bn': 3, 'fec': 0, 'pf6': 1}, 13),
        ({'bn': 4}, 260),
    ],
)
def test_speciation_find_clusters(cluster, n_clusters, solvation_data):
    speciation = Speciation(solvation_data, 10, 49)
    df = speciation.find_shells(cluster)
    assert len(df) == n_clusters


@pytest.mark.parametrize(
    "solvent_one, solvent_two, correlation",
    [
        ('bn', 'bn', 0.98),
        ('fec', 'bn', 1.03),
        ('fec', 'pf6', 0.15),
    ],
)
def test_speciation_correlation(solvent_one, solvent_two, correlation, solvation_data):
    speciation = Speciation(solvation_data, 10, 49)
    df = speciation.co_occurrence
    np.testing.assert_allclose(df[solvent_one][solvent_two], correlation, atol=0.05)


def test_plot_correlation(solvation_data):
    speciation = Speciation(solvation_data, 10, 49)
    fig, ax = speciation.plot_co_occurrence()
    # fig.show()


@pytest.mark.parametrize(
    "name, cn",
    [
        ("fec", 0.25),
        ("bn", 4.33),
        ("pf6", 0.15),
    ],
)
def test_coordination(name, cn, solvation_data, run_solution):
    atoms = run_solution.u.atoms
    coordination = Coordination(solvation_data, 10, 49, atoms)
    np.testing.assert_allclose(cn, coordination.cn_dict[name], atol=0.05)
    assert len(coordination.cn_by_frame) == 3


@pytest.mark.parametrize(
    "name, atom_type, percent",
    [
        ("fec", 19, 0.008),
        ("bn", 5, 0.9976),
        ("pf6", 21, 1.000),
    ],
)
def test_coordinating_atoms(name, atom_type, percent, solvation_data, run_solution):
    atoms = run_solution.u.atoms
    coordination = Coordination(solvation_data, 10, 49, atoms)
    calculated_percent = coordination.coordinating_atoms.loc[(name, atom_type)]
    np.testing.assert_allclose(percent, calculated_percent, atol=0.05)


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
    np.testing.assert_allclose([percent], pairing.pairing_dict[name], atol=0.05)
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
    np.testing.assert_allclose([percent], pairing.percent_free_solvents[name], atol=0.05)
