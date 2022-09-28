import matplotlib.pyplot as plt
import warnings
import pytest
from solvation_analysis.solute import Solute
import numpy as np
from MDAnalysis import Universe

from solvation_analysis.tests.conftest import u_eax_series, u_eax_atom_groups


def test_instantiate_solute(pre_solute):
    # these check basic properties of the instantiation
    assert len(pre_solute.radii) == 3
    assert callable(pre_solute.kernel)
    assert pre_solute.solute.n_residues == 49
    assert pre_solute.solvents['pf6'].n_residues == 49
    assert pre_solute.solvents['fec'].n_residues == 237
    assert pre_solute.solvents['bn'].n_residues == 363


def test_networking_instantiation_error(atom_groups):
    li = atom_groups['li']
    pf6 = atom_groups['pf6']
    bn = atom_groups['bn']
    fec = atom_groups['fec']
    with pytest.raises(Exception):
        solute = Solute(
            li, {'pf6': pf6, 'bn': bn, 'fec': fec}, analysis_classes=['networking']
        )


def test_plot_solvation_distance(rdf_bins_and_data_easy):
    bins, data = rdf_bins_and_data_easy['pf6_all']
    fig, ax = Solute._plot_solvation_radius(bins, data, 2)
    # fig.show()  # comment out for global testing


def test_radii_finding(run_solute):
    # checks that the solvation radii are plotted
    assert len(run_solute.radii) == 3
    assert len(run_solute.rdf_data) == 3
    # checks that the identified solvation radii are approximately correct
    assert 2 < run_solute.radii['pf6'] < 3
    assert 2 < run_solute.radii['fec'] < 3
    assert 2 < run_solute.radii['bn'] < 3
    # for fig, ax in run_solute.rdf_plots.values():
    # plt.show()  # comment out for global testing


def test_run_warning(pre_solute_mutable):
    # checks that an error is thrown if there are not enough radii
    pre_solute_mutable.radii = {'pf6': 2.8}
    with pytest.raises(AssertionError):
        pre_solute_mutable.run(step=1)


def test_run(pre_solute_mutable):
    # checks that run is run correctly
    pre_solute_mutable.run(step=1)
    assert len(pre_solute_mutable.solvation_frames) == 10
    assert len(pre_solute_mutable.solvation_frames[0]) == 228
    assert len(pre_solute_mutable.solvation_data) == 2312


def test_run_w_all(pre_solute_mutable):
    # checks that run is run correctly
    pre_solute_mutable.analysis_classes = [
        "pairing", "coordination", "speciation", "residence", "networking"
    ]
    pre_solute_mutable.networking_solvents = 'pf6'
    pre_solute_mutable.run(step=1)
    assert len(pre_solute_mutable.solvation_frames) == 10
    assert len(pre_solute_mutable.solvation_frames[0]) == 228
    assert len(pre_solute_mutable.solvation_data) == 2312


@pytest.mark.parametrize(
    "solute_index, radius, frame, expected_res_ids",
    [
        (1, 3, 5, [46, 100, 171, 255, 325, 521, 650]),
        (2, 3, 6, [13, 59, 177, 264, 314, 651]),
        (40, 3.5, 0, [101, 126, 127, 360, 368, 305, 689])
    ],
)
def test_radial_shell(solute_index, radius, frame, expected_res_ids, run_solute):
    run_solute.u.trajectory[frame]
    shell = run_solute.radial_shell(solute_index, radius)
    assert set(shell.resindices) == set(expected_res_ids)


@pytest.mark.parametrize(
    "solute_index, n_mol, frame, expected_res_ids",
    [
        (6741, 4, 5, [46, 100, 171, 255, 650]),
        (6749, 5, 6, [13, 59, 177, 264, 314, 651]),
        (7053, 6, 0, [101, 126, 127, 360, 368, 305, 689])
    ],
)
def test_closest_n_mol(solute_index, n_mol, frame, expected_res_ids, run_solute):
    run_solute.u.trajectory[frame]
    shell = run_solute.closest_n_mol(solute_index, n_mol)
    assert set(shell.resindices) == set(expected_res_ids)


@pytest.mark.parametrize(
    "solute_index, step, expected_res_ids",
    [
        (6741, 5, [46, 100, 171, 255, 650]),
        (6749, 6, [13, 59, 177, 264, 314, 651]),
        (7053, 0, [101, 126, 127, 360, 689])
    ],
)
def test_solvation_shell(solute_index, step, expected_res_ids, run_solute):
    shell = run_solute.solvation_shell(solute_index, step)
    assert set(shell.resindices) == set(expected_res_ids)


@pytest.mark.parametrize(
    "solute_index, step, remove, expected_res_ids",
    [
        (6741, 5, {'bn': 1}, [46, 171, 255, 650]),
        (6749, 6, {'bn': 2, 'fec': 1}, [13, 177, 314, 651]),
        (7053, 0, {'fec': 1}, [101, 126, 127, 360, 689])
    ],
)
def test_solvation_shell_remove_mols(solute_index, step, remove, expected_res_ids, run_solute):
    shell = run_solute.solvation_shell(solute_index, step, remove_mols=remove)
    assert set(shell.resindices) == set(expected_res_ids)


@pytest.mark.parametrize(
    "solute_index, step, n, expected_res_ids",
    [
        (6741, 5, 3, [46, 171, 255, 650]),
        (6749, 6, 3, [13, 177, 314, 651]),
        (7053, 0, 4, [101, 126, 127, 360, 689]),
        (7053, 0, 1, [101, 689])
    ],
)
def test_solvation_shell_remove_closest(solute_index, step, n, expected_res_ids, run_solute):
    shell = run_solute.solvation_shell(solute_index, step, closest_n_only=n)
    assert set(shell.resindices) == set(expected_res_ids)


@pytest.mark.parametrize(
    "shell, n_shells",
    [
        ({'bn': 5, 'fec': 0, 'pf6': 0}, 175),
        ({'bn': 3, 'fec': 3, 'pf6': 0}, 2),
        ({'bn': 3, 'fec': 0, 'pf6': 1}, 13),
        ({'bn': 4}, 260),
    ],
)
def test_speciation_find_shells(shell, n_shells, run_solute):
    # duplicated to test in solute
    df = run_solute.speciation.find_shells(shell)
    assert len(df) == n_shells


@pytest.mark.parametrize(
    "name, cn",
    [
        ("fec", 0.25),
        ("bn", 4.33),
        ("pf6", 0.15),
    ],
)
def test_coordination_numbers(name, cn, run_solute):
    # duplicated to test in solute
    coord_dict = run_solute.coordination.cn_dict
    np.testing.assert_allclose(cn, coord_dict[name], atol=0.05)


@pytest.mark.parametrize(
    "name, percent",
    [
        ("fec", 0.21),
        ("bn", 1.0),
        ("pf6", 0.14),
    ],
)
def test_pairing(name, percent, run_solute):
    # duplicated to test in solute
    pairing_dict = run_solute.pairing.pairing_dict
    np.testing.assert_allclose([percent], pairing_dict[name], atol=0.05)


@pytest.mark.parametrize("name", ['ea', 'eaf', 'fea', 'feaf'])
def test_instantiate_eax_solvents(name, u_eax_series):
    assert isinstance(u_eax_series[name], Universe)


@pytest.mark.parametrize("name", ['ea', 'eaf', 'fea', 'feaf'])
def test_instantiate_eax_atom_groups(name, u_eax_atom_groups):
    all_atoms = len(u_eax_atom_groups[name]['li'].universe.atoms)
    all_atoms_in_groups = sum([len(ag) for ag in u_eax_atom_groups[name].values()])
    assert all_atoms_in_groups == all_atoms


@pytest.mark.parametrize("name", ['ea', 'eaf', 'fea', 'feaf'])
def test_instantiate_eax_solutes(name, eax_solutes):
    assert isinstance(eax_solutes[name], Solute)


def test_iba_atom_groups(iba_atom_groups):
    n_atoms = len(iba_atom_groups['iba'].universe.atoms)
    group_names = [
        'h2o_O', 'h2o_H', 'iba_alcohol_O', 'iba_alcohol_H', 'iba_ketone', 'iba_C', 'iba_C_H'
    ]
    group_lengths = [len(iba_atom_groups[name]) for name in group_names]
    assert sum(group_lengths) == n_atoms


def test_iba_solutes(iba_solute):
    assert isinstance(iba_solute, Solute)
