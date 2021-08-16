import matplotlib.pyplot as plt
import warnings
import pytest
from solvation_analysis.solution import Solution
import numpy as np


def test_instantiate_solute(pre_solution):
    # these check basic properties of the instantiation
    assert len(pre_solution.radii) == 1
    assert callable(pre_solution.kernel)
    assert pre_solution.solute.n_residues == 49
    assert pre_solution.solvents['pf6'].n_residues == 49
    assert pre_solution.solvents['fec'].n_residues == 237
    assert pre_solution.solvents['bn'].n_residues == 363


def test_plot_solvation_distance(rdf_bins_and_data_easy):
    bins, data = rdf_bins_and_data_easy['pf6_all']
    fig, ax = Solution._plot_solvation_radius(bins, data, 2)
    # plt.show()  # comment out for global testing


def test_radii_finding(run_solution):
    # checks that the solvation radii are plotted
    assert len(run_solution.radii) == 3
    assert len(run_solution.rdf_data) == 3
    # checks that the identified solvation radii are approximately correct
    assert 2 < run_solution.radii['pf6'] < 3
    assert 2 < run_solution.radii['fec'] < 3
    assert 2 < run_solution.radii['bn'] < 3
    # for fig, ax in run_solute.rdf_plots.values():
    #     plt.show()  # comment out for global testing


def test_run_warning(pre_solution_mutable):
    # checks that an error is thrown if there are not enough radii
    with pytest.raises(AssertionError):
        pre_solution_mutable.run(step=1)


def test_run(pre_solution_mutable):
    # checks that run is run correctly
    pre_solution_mutable.radii = {'pf6': 2.8}
    pre_solution_mutable.run(step=1)
    assert len(pre_solution_mutable.solvation_frames) == 10
    assert len(pre_solution_mutable.solvation_frames[0]) == 228
    assert len(pre_solution_mutable.solvation_data) == 2317


@pytest.mark.parametrize(
    "solute_index, radius, frame, expected_res_ids",
    [
        (1, 3, 5, [47, 101, 172, 256, 326, 522, 652]),
        (2, 3, 6, [14, 60, 178, 265, 315, 653]),
        (40, 3.5, 0, [102, 127, 128, 361, 369, 306, 691])
    ],
)
def test_radial_shell(solute_index, radius, frame, expected_res_ids, run_solution):
    run_solution.u.trajectory[frame]
    shell = run_solution.radial_shell(solute_index, radius)
    assert set(shell.resids) == set(expected_res_ids)


@pytest.mark.parametrize(
    "solute_index, n_mol, frame, expected_res_ids",
    [
        (1, 4, 5, [47, 101, 172, 256, 652]),
        (2, 5, 6, [14, 60, 178, 265, 315, 653]),
        (40, 6, 0, [102, 127, 128, 361, 369, 306, 691])
    ],
)
def test_closest_n_mol(solute_index, n_mol, frame, expected_res_ids, run_solution):
    run_solution.u.trajectory[frame]
    shell = run_solution.closest_n_mol(solute_index, n_mol)
    assert set(shell.resids) == set(expected_res_ids)


@pytest.mark.parametrize(
    "solute_index, step, expected_res_ids",
    [
        (1, 5, [47, 101, 172, 256, 652]),
        (2, 6, [14, 60, 178, 265, 315, 653]),
        (40, 0, [102, 127, 128, 361, 691])
    ],
)
def test_solvation_shell(solute_index, step, expected_res_ids, run_solution):
    shell = run_solution.solvation_shell(solute_index, step)
    assert set(shell.resids) == set(expected_res_ids)


@pytest.mark.parametrize(
    "solute_index, step, remove, expected_res_ids",
    [
        (1, 5, {'bn': 1}, [47, 172, 256, 652]),
        (2, 6, {'bn': 2, 'fec': 1}, [14, 178, 315, 653]),
        (40, 0, {'fec': 1}, [102, 127, 128, 361, 691])
    ],
)
def test_solvation_shell_remove(solute_index, step, remove, expected_res_ids, run_solution):
    shell = run_solution.solvation_shell(solute_index, step, remove_mols=remove)
    assert set(shell.resids) == set(expected_res_ids)


@pytest.mark.parametrize(
    "solute_index, step, remove, expected_res_ids",
    [
        (1, 5, {'bn': 1}, [47, 172, 256, 652]),
        (2, 6, {'bn': 2, 'fec': 1}, [14, 178, 315, 653]),
        (40, 0, {'fec': 1}, [102, 127, 128, 361, 691])
    ],
)
def test_solvation_shell_remove(solute_index, step, remove, expected_res_ids, run_solution):
    shell = run_solution.solvation_shell(solute_index, step, remove_mols=remove)
    assert set(shell.resids) == set(expected_res_ids)


@pytest.mark.parametrize(
    "solute_index, step, n, expected_res_ids",
    [
        (1, 5, 3, [47, 172, 256, 652]),
        (2, 6, 3, [14, 178, 315, 653]),
        (40, 0, 4, [102, 127, 128, 361, 691]),
        (40, 0, 1, [102, 691])
    ],
)
def test_solvation_shell_remove(solute_index, step, n, expected_res_ids, run_solution):
    shell = run_solution.solvation_shell(solute_index, step, closest_n_only=n)
    assert set(shell.resids) == set(expected_res_ids)


@pytest.mark.parametrize(
    "cluster, n_clusters",
    [
        ({'bn': 5, 'fec': 0, 'pf6': 0}, 175),
        ({'bn': 3, 'fec': 3, 'pf6': 0}, 2),
        ({'bn': 3, 'fec': 0, 'pf6': 1}, 8),
        ({'bn': 4}, 260),
    ],
)
def test_speciation_find_clusters(cluster, n_clusters, run_solution):
    df = run_solution.speciation.find_shells(cluster)
    assert len(df) == n_clusters


@pytest.mark.parametrize(
    "name, cn",
    [
        ("fec", 0.25),
        ("bn", 4.33),
        ("pf6", 0.15),
    ],
)
def test_coordination_numbers(name, cn, run_solution):
    coord_dict = run_solution.coordination.cn_dict
    np.testing.assert_allclose(cn, coord_dict[name], atol=0.05)


@pytest.mark.parametrize(
    "name, percent",
    [
        ("fec", 0.21),
        ("bn", 1.0),
        ("pf6", 0.14),
    ],
)
def test_pairing(name, percent, run_solution):
    pairing_dict = run_solution.pairing.pairing_dict
    np.testing.assert_allclose([percent], pairing_dict[name], atol=0.05)
