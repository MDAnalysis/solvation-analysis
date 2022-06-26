import numpy as np
import pandas as pd
import pytest

from solvation_analysis.analysis_library import (
    Speciation,
    Coordination,
    Pairing,
    Residence,
    Networking,
)


@pytest.mark.parametrize(
    "shell, percent",
    [
        ({'bn': 5, 'fec': 0, 'pf6': 0}, 0.357),
        ({'bn': 3, 'fec': 3, 'pf6': 0}, 0.004),
        ({'bn': 3, 'fec': 0, 'pf6': 1}, 0.016),
        ({'bn': 4}, 0.531),
    ],
)
def test_speciation_shell_percent(shell, percent, solvation_data):
    speciation = Speciation(solvation_data, 10, 49)
    percentage = speciation.shell_percent(shell)
    np.testing.assert_allclose(percent, percentage, atol=0.05)


@pytest.mark.parametrize(
    "shell, n_shells",
    [
        ({'bn': 5, 'fec': 0, 'pf6': 0}, 175),
        ({'bn': 3, 'fec': 3, 'pf6': 0}, 2),
        ({'bn': 3, 'fec': 0, 'pf6': 1}, 13),
        ({'bn': 4}, 260),
    ],
)
def test_speciation_find_shells(shell, n_shells, solvation_data):
    speciation = Speciation(solvation_data, 10, 49)
    df = speciation.find_shells(shell)
    assert len(df) == n_shells


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
        ("fec", '19', 0.008),
        ("bn", '5', 0.9976),
        ("pf6", '21', 1.000),
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
    np.testing.assert_allclose(diluent_percent, pairing.diluent_composition[name], atol=0.05)
    np.testing.assert_allclose(sum(pairing.diluent_composition.values()), 1, atol=0.05)


def test_from_solution(run_solution):
    residence = Residence.from_solution(run_solution)
    assert len(residence.residence_times) == 3
    assert len(residence.residence_times_fit) == 3


def test_residence_times(solvation_data_sparse):
    residence = Residence(solvation_data_sparse, step=10)
    times = residence.residence_times
    # we do not use a parameterization because this operation is expensive
    np.testing.assert_almost_equal(times['fec'], 10, 3)
    np.testing.assert_almost_equal(times['bn'], 80, 3)
    np.testing.assert_almost_equal(times['pf6'], np.nan, 3)


def test_plot_auto_covariance(solvation_data_sparse):
    residence = Residence(solvation_data_sparse, step=10)
    # we do not use a parameterization because this operation is expensive
    residence.plot_auto_covariance('fec')
    residence.plot_auto_covariance('bn')
    residence.plot_auto_covariance('pf6')


def test_residence_time_warning(solvation_data_sparse):
    # we step through the data frame to speed up the tests
    with pytest.warns(UserWarning, match="the autocovariance for pf6 does not converge"):
        Residence(solvation_data_sparse, step=10)


def test_network_finder(run_solution):
    networking = Networking.from_solution(run_solution, ['pf6'])
    network_df = networking.network_df
    assert len(network_df) == 128
    # TODO: implement real testing
    res_ix = networking.get_cluster_res_ix(0, 0)
    run_solution.u.residues[res_ix.astype(int)].atoms
    return


def test_timing_benchmark(solvation_data_large):
    """
    # total timing of 1.07 seconds!!! wooo!!!
    # not bad!
    """
    import time
    start = time.time()
    # uncomment these to perform a benchmarking
    # residence = Residence(solvation_data_large, step=1)
    # times = residence.residence_times
    total_time = time.time() - start
    return
