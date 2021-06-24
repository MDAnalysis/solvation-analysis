"""
Unit and regression test for the solvation_analysis package.
"""


# Import package, test suite, and other packages as needed
import MDAnalysis as mda
import numpy as np
import pytest

from solvation_analysis.solvation import (
    get_atom_group,
    get_closest_n_mol,
    get_radial_shell,
)

from solvation_analysis.tests.test_basics import (
    u_grid_1,
    u_grid_3,
    u_real,
    u_real_named,
    atom_groups,
)


def test_get_atom_group(u_real_named):
    # test that atoms, residues, and groups are being correctly converted to AtomGroup
    u = u_real_named
    res_group = u.residues[1:5]
    atom_group = u.atoms[1:5]
    res = u.residues[1]
    atom = u.atoms[1]
    groups = [res_group, atom_group, res, atom]
    for group in groups:
        assert isinstance(get_atom_group(u, group), mda.core.groups.AtomGroup)


def test_get_closest_n_mol_correct_number(u_grid_1, u_real, atom_groups):
    # test that the correct number of residues are being returned
    test_li = atom_groups["li"][0]
    shell_sizes = range(2, 8)
    # test on real system
    for size in shell_sizes:
        shell = get_closest_n_mol(u_real, test_li, n_mol=size)
        assert len(shell.residues) == size + 1

    test_atoms = u_grid_1.atoms[[0, 3, 10, 16, 42]]
    # test on grid
    for test_atom in test_atoms:
        atoms = get_closest_n_mol(u_grid_1, test_atom, n_mol=5)
        assert len(atoms) == 6


def test_get_closest_n_mol_correct_ids(u_grid_1, u_real):
    # test that the correct atoms are being returned
    # test on grid
    expected_shell_ids = {10: [5, 10, 11, 12, 17, 47],
                          16: [11, 16, 17, 18, 23, 53],
                          42: [7, 37, 43, 44, 49, 79]}
    test_atoms = u_grid_1.atoms[[10, 16, 42]]  # only atoms on side of box
    for test_atom in test_atoms:
        expected_ids = expected_shell_ids[test_atom.ix]
        shell_ids = get_closest_n_mol(u_grid_1, test_atom, n_mol=5).resids
        np.testing.assert_array_equal(shell_ids, expected_ids)


def test_get_closest_n_mol_radii_invariance(u_real, atom_groups):
    # test that the return_radii does not effect behavior
    # test on real system
    test_li = atom_groups["li"][0]
    radii_range = range(0, 5)
    default_shell, default_resids, default_radii = get_closest_n_mol(
        u_real, test_li, return_resids=True, return_radii=True
    )
    for rad in radii_range:
        shell, resids, radii = get_closest_n_mol(
            u_real, test_li, radius=rad, return_resids=True, return_radii=True
        )
        assert shell == default_shell
        np.testing.assert_allclose(resids, default_resids)
        np.testing.assert_allclose(radii, default_radii)


def test_get_radial_shell_correct_number(u_grid_1, u_real, atom_groups):
    # test that the correct number of atoms are being returned
    # test on the grid
    test_atoms = u_grid_1.atoms[[0, 3, 10, 44]]  # corner, edge, side, center
    distances = [0.95, 1.05, 1.45]
    expected_shell_sizes = {
        0.95: [1, 1, 1, 1],  # no atoms within 1 A of test atoms because grid size is 1
        1.05: [4, 5, 6, 7],  # adjacent atoms for corner, edge, side, and center atoms
        1.45: [7, 10, 14, 19],  # adjacent atoms + diagonally adjacent atoms for ^^
    }
    for distance in distances:
        expected_sizes = expected_shell_sizes[distance]
        shell_sizes = [len(get_radial_shell(u_grid_1, atom, distance)) for atom in test_atoms]
        np.testing.assert_allclose(expected_sizes, shell_sizes)
    # test on real system
    test_li = atom_groups["li"][0]
    radii_range = range(2, 8)
    shell_sizes = [1, 59, 81, 123, 142, 191]
    for rad, size in zip(radii_range, shell_sizes):
        assert size == len(get_radial_shell(u_real, test_li, radius=rad))
    assert len(get_radial_shell(u_real, test_li, radius=100)) == len(u_real.atoms)


def test_get_radial_shell_correct_ids(u_grid_1):
    # test that the correct ids are being retured
    # test on grid
    expected_shell_ids = {10: [5, 10, 11, 12, 17, 47],
                          16: [11, 16, 17, 18, 23, 53],
                          42: [7, 37, 43, 44, 49, 79]}
    test_atoms = u_grid_1.atoms[[10, 16, 42]]  # only atoms on side of box
    for test_atom in test_atoms:
        expected_ids = expected_shell_ids[test_atom.ix]
        shell_ids = get_radial_shell(u_grid_1, test_atom, 1.05).resids
        np.testing.assert_array_equal(shell_ids, expected_ids)
