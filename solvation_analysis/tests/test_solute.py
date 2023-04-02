from functools import reduce

import matplotlib.pyplot as plt
import warnings
import pytest
from solvation_analysis.solute import Solute
import numpy as np
from MDAnalysis import Universe

from solvation_analysis.tests.conftest import u_eax_series, u_eax_atom_groups


def test_instantiate_solute_from_atoms(pre_solute):
    # these check basic properties of the instantiation
    assert len(pre_solute.radii) == 3
    assert callable(pre_solute.kernel)
    assert pre_solute.solute_atoms.n_residues == 49
    assert pre_solute.solvents['pf6'].n_residues == 49
    assert pre_solute.solvents['fec'].n_residues == 237
    assert pre_solute.solvents['bn'].n_residues == 363


def test_init_fail(atom_groups):
    with pytest.raises(RuntimeError):
        Solute(atom_groups['li'], {'pf6': atom_groups['pf6']})


def test_networking_instantiation_error(atom_groups):
    li = atom_groups['li']
    pf6 = atom_groups['pf6']
    bn = atom_groups['bn']
    fec = atom_groups['fec']
    with pytest.raises(Exception):
        Solute.from_atoms(
            li, {'pf6': pf6, 'bn': bn, 'fec': fec}, analysis_classes=['networking']
        )


def test_plot_solvation_distance(rdf_bins_and_data_easy):
    bins, data = rdf_bins_and_data_easy['pf6_all']
    fig, ax = Solute._plot_solvation_radius(bins, data, 2)
    # fig.show()  # comment out for global testing


def test_radii_finding(run_solute):
    # checks that the solvation radii are plotted
    assert len(run_solute.radii) == 3
    assert len(run_solute.rdf_data["solute_0"]) == 3
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
    assert len(pre_solute_mutable._solvation_frames) == 10
    assert len(pre_solute_mutable._solvation_frames[0]) == 228
    assert len(pre_solute_mutable.solvation_data) == 2312


def test_run_w_all(pre_solute_mutable):
    # checks that run is run correctly
    pre_solute_mutable.analysis_classes = [
        "pairing", "coordination", "speciation", "residence", "networking"
    ]
    pre_solute_mutable.networking_solvents = 'pf6'
    pre_solute_mutable.run(step=1)
    assert len(pre_solute_mutable._solvation_frames) == 10
    assert len(pre_solute_mutable._solvation_frames[0]) == 228
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
    shell = run_solute.get_closest_n_mol(solute_index, n_mol)
    assert set(shell.resindices) == set(expected_res_ids)


@pytest.mark.parametrize(
    "solute_index, step, expected_res_ids",
    [
        (650, 5, [46, 100, 171, 255, 650]),
        (651, 6, [13, 59, 177, 264, 314, 651]),
        (689, 0, [101, 126, 127, 360, 689])
    ],
)
def test_solvation_shell(solute_index, step, expected_res_ids, run_solute):
    # TODO: something is broken in the tutorial here
    shell = run_solute.get_shell(solute_index, step)
    assert set(shell.resindices) == set(expected_res_ids)


@pytest.mark.parametrize(
    "solute_index, step, remove, expected_res_ids",
    [
        (650, 5, {'bn': 1}, [46, 171, 255, 650]),
        (651, 6, {'bn': 2, 'fec': 1}, [13, 177, 314, 651]),
        (689, 0, {'fec': 1}, [101, 126, 127, 360, 689])
    ],
)
def test_solvation_shell_remove_mols(solute_index, step, remove, expected_res_ids, run_solute):
    shell = run_solute.get_shell(solute_index, step, remove_mols=remove)
    assert set(shell.resindices) == set(expected_res_ids)


@pytest.mark.parametrize(
    "solute_index, step, n, expected_res_ids",
    [
        (650, 5, 3, [46, 171, 255, 650]),
        (651, 6, 3, [13, 177, 314, 651]),
        (689, 0, 4, [101, 126, 127, 360, 689]),
        (689, 0, 1, [101, 689])
    ],
)
def test_solvation_shell_remove_closest(solute_index, step, n, expected_res_ids, run_solute):
    shell = run_solute.get_shell(solute_index, step, closest_n_only=n)
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
    df = run_solute.speciation.get_shells(shell)
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
    coord_dict = run_solute.coordination.coordination_numbers
    np.testing.assert_allclose(cn, coord_dict[name], atol=0.05)


@pytest.mark.parametrize(
    "name, fraction",
    [
        ("fec", 0.21),
        ("bn", 1.0),
        ("pf6", 0.14),
    ],
)
def test_pairing(name, fraction, run_solute):
    # duplicated to test in solute
    pairing_dict = run_solute.pairing.solvent_pairing
    np.testing.assert_allclose([fraction], pairing_dict[name], atol=0.05)


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


def test_plot_solvation_radius(run_solute, iba_small_solute):
    run_solute.plot_solvation_radius('solute_0', 'fec')
    iba_small_solute.plot_solvation_radius('iba_ketone', 'iba')


@pytest.mark.parametrize("residue", ['iba_ketone', 'solute', 'H2O', 'iba'])
def test_draw_molecule_string(iba_solutes, residue):
    iba_solutes['iba_ketone'].draw_molecule(residue)


def test_draw_molecule_residue(iba_solutes):
    solute = iba_solutes['iba_ketone']
    residue = solute.u.atoms.residues[0]
    solute.draw_molecule(residue)


def test_iba_solutes(iba_solutes):
    for solute in iba_solutes.values():
        assert isinstance(solute, Solute)


def test_from_atoms(iba_atom_groups, iba_solvents):
    solute_atoms = (
            iba_atom_groups['iba_ketone'] +
            iba_atom_groups['iba_alcohol_O'] +
            iba_atom_groups['iba_alcohol_H']
    )
    solute = Solute.from_atoms(solute_atoms, iba_solvents)
    solute.run()
    assert set(solute.atom_solutes.keys()) == {'solute_0', 'solute_1', 'solute_2'}


def test_from_atoms_errors(iba_atom_groups, H2O_atom_groups, iba_solvents):
    solute_atoms = (
            iba_atom_groups['iba_ketone'] +
            iba_atom_groups['iba_alcohol_O'] +
            iba_atom_groups['iba_alcohol_H']
    )
    with pytest.raises(AssertionError):
        bad_atoms = solute_atoms[:-2]
        Solute.from_atoms(bad_atoms, iba_solvents)

    with pytest.raises(AssertionError):
        bad_atoms = solute_atoms + H2O_atom_groups['H2O_O']
        Solute.from_atoms(bad_atoms, iba_solvents)


def test_from_atoms_dict(iba_atom_groups, iba_solvents):
    solute_atoms = {
        'iba_ketone': iba_atom_groups['iba_ketone'],
        'iba_alcohol_O': iba_atom_groups['iba_alcohol_O'],
        'iba_alcohol_H': iba_atom_groups['iba_alcohol_H']
    }
    solute = Solute.from_atoms_dict(solute_atoms, iba_solvents)
    assert set(solute.atom_solutes.keys()) == {'iba_ketone', 'iba_alcohol_O', 'iba_alcohol_H'}
    solute.run()


def test_from_atoms_dict_errors(iba_atom_groups, H2O_atom_groups, iba_solvents):
    solute_atoms = {
        'iba_ketone': iba_atom_groups['iba_ketone'],
        'iba_alcohol_O': iba_atom_groups['iba_alcohol_O'],
        'iba_alcohol_H': iba_atom_groups['iba_alcohol_H']
    }
    with pytest.raises(AssertionError):
        bad_atoms = {**solute_atoms}
        bad_atoms['iba_ketone'] = bad_atoms['iba_ketone'][:-2]
        Solute.from_atoms_dict(bad_atoms, iba_solvents)

    with pytest.raises(AssertionError):
        bad_atoms = {**solute_atoms}
        bad_atoms['iba_ketone'] = bad_atoms['iba_ketone'] + bad_atoms['iba_alcohol_O']
        Solute.from_atoms_dict(bad_atoms, iba_solvents)

    with pytest.raises(AssertionError):
        bad_atoms = {**solute_atoms}
        bad_atoms['iba_ketone'] = bad_atoms['iba_alcohol_O']
        Solute.from_atoms_dict(bad_atoms, iba_solvents)

    with pytest.raises(AssertionError):
        bad_atoms = {**solute_atoms}
        bad_atoms['H2O_O'] = H2O_atom_groups['H2O_O']
        Solute.from_atoms_dict(bad_atoms, iba_solvents)


def test_from_solute_list(iba_solutes, iba_solvents):
    solute_list = [
            iba_solutes['iba_ketone'],
            iba_solutes['iba_alcohol_O'],
            iba_solutes['iba_alcohol_H']
    ]
    solute = Solute.from_solute_list(solute_list, iba_solvents)
    solute.run()
    assert set(solute.atom_solutes.keys()) == {'iba_ketone', 'iba_alcohol_O', 'iba_alcohol_H'}


def test_from_solute_list_restepped(iba_solutes, iba_atom_groups, iba_solvents):
    new_solvent = {"H2O": iba_solvents["H2O"]}
    new_ketone = Solute.from_atoms(
        iba_atom_groups['iba_ketone'],
        new_solvent,
        solute_name='iba_ketone'
    )
    new_ketone.run(step=2)
    solute_list = [iba_solutes['iba_alcohol_O'], new_ketone]
    solute = Solute.from_solute_list(solute_list, iba_solvents)
    with pytest.warns(UserWarning, match='re-run') as record:
        solute.run(step=2)
        user_warnings = 0
        for warning in record:
            if warning.category == UserWarning:
                user_warnings += 1
        assert user_warnings == 2
    assert set(solute.atom_solutes.keys()) == {'iba_ketone', 'iba_alcohol_O'}


def test_from_solute_list_errors(iba_solutes, H2O_atom_groups, iba_solvents):
    solute_list = [
        iba_solutes['iba_ketone'],
        iba_solutes['iba_alcohol_O'],
        iba_solutes['iba_alcohol_H']
    ]

    H2O_solute = Solute.from_atoms(H2O_atom_groups['H2O_O'], iba_solvents)
    with pytest.raises(AssertionError):
        bad_solute_list = [*solute_list]
        bad_solute_list.append(H2O_solute)
        Solute.from_solute_list(bad_solute_list, iba_solvents)

    iba_ketone_renamed = Solute.from_atoms(
        iba_solutes['iba_ketone'].solute_atoms,
        iba_solvents,
        solute_name='iba_alcohol_O'
    )
    with pytest.raises(AssertionError):
        bad_solute_list = [*solute_list]
        bad_solute_list[0] = iba_ketone_renamed
        Solute.from_solute_list(bad_solute_list, iba_solvents)

    with pytest.raises(AssertionError):
        bad_solute_list = [1, 2, 3]
        Solute.from_solute_list(bad_solute_list, iba_solvents)


def test_iba_all_analysis(iba_atom_groups, iba_solvents):
    solute_atoms = reduce(lambda x, y: x | y, [solute for solute in iba_atom_groups.values()])
    solute = Solute.from_atoms(
        solute_atoms,
        iba_solvents,
        networking_solvents=['iba'],
        analysis_classes='all',
        radii={'iba': 1.9, 'H2O': 1.9},
    )
    # TODO: get this passing
    solute.run(step=4)
    return
