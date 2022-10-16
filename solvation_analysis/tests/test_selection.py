"""
Unit and regression test for the solvation_analysis package.
"""


# Import package, test suite, and other packages as needed
import MDAnalysis as mda
import numpy as np
import pytest

from solvation_analysis._utils import get_atom_group, get_closest_n_mol, get_radial_shell


def test_get_atom_group(u_real_named):
    # test that atoms, residues, and groups are being correctly converted to AtomGroup
    u = u_real_named
    res_group = u.residues[1:5]
    atom_group = u.atoms[1:5]
    res = u.residues[1]
    atom = u.atoms[1]
    groups = [res_group, atom_group, res, atom]
    for group in groups:
        assert isinstance(get_atom_group(group), mda.core.groups.AtomGroup)


@pytest.mark.parametrize("shell_size", [2, 3, 4, 5, 6, 7])
def test_get_closest_n_mol_correct_number_real(shell_size, u_real, atom_groups):
    # test that the correct number of residues are being returned, on real system
    test_li = atom_groups["li"][0]
    shell = get_closest_n_mol(test_li, n_mol=shell_size)
    assert len(shell.residues) == shell_size + 1


@pytest.mark.parametrize("test_ix", [0, 3, 10, 16, 42])
def test_get_closest_n_mol_correct_number_grid(test_ix, u_grid_1):
    # test that the correct number of residues are being returned, on grid system
    test_atom = u_grid_1.atoms[test_ix]
    atoms = get_closest_n_mol(test_atom, n_mol=5)
    assert len(atoms) == 6


@pytest.mark.parametrize(
    "center_ix, expected_ix",
    [
        (10, [4, 9, 10, 11, 16, 46]),
        (16, [10, 15, 16, 17, 22, 52]),
        (42, [6, 36, 42, 43, 48, 78]),
    ],
)
def test_get_closest_n_mol_correct_ix(center_ix, expected_ix, u_grid_1):
    # test that the correct atoms are being returned, on grid system
    test_atom = u_grid_1.atoms[center_ix]  # only atoms on side of box
    shell_ix = get_closest_n_mol(test_atom, n_mol=5).resindices
    np.testing.assert_array_equal(shell_ix, expected_ix)


@pytest.mark.parametrize("radius", [0, 1, 2, 3, 4, 5])
def test_get_closest_n_mol_radii_invariance(radius, u_real, atom_groups):
    # test that the return_radii does not effect behavior, on real system
    test_li = atom_groups["li"][0]
    default_shell, default_resix, default_radii = get_closest_n_mol(
        test_li, 5, return_ordered_resix=True, return_radii=True
    )
    shell, resix, radii = get_closest_n_mol(
        test_li, 5, guess_radius=radius, return_ordered_resix=True, return_radii=True
    )
    assert shell == default_shell
    np.testing.assert_allclose(resix, default_resix)
    np.testing.assert_allclose(radii, default_radii)


@pytest.mark.parametrize(
    "distance, expected_sizes",
    [(0.95, [1, 1, 1, 1]), (1.05, [4, 5, 6, 7]), (1.45, [7, 10, 14, 19])],
)
def test_get_radial_shell_correct_number_grid(distance, expected_sizes, u_grid_1):
    # test that the correct shell sizes are being returned, on grid system
    test_atoms = u_grid_1.atoms[[0, 3, 10, 44]]  # corner, edge, side, center
    shell_sizes = [len(get_radial_shell(atom, distance)) for atom in test_atoms]
    np.testing.assert_allclose(expected_sizes, shell_sizes)


@pytest.mark.parametrize(
    "radius, shell_size", [(2, 1), (3, 59), (4, 81), (5, 123), (6, 142), (7, 191)]
)
def test_get_radial_shell_correct_number_real(radius, shell_size, u_real, atom_groups):
    # test that the correct sizes are being returned, on real system
    test_li = atom_groups["li"][0]
    assert shell_size == len(get_radial_shell(test_li, radius=radius))
    assert len(get_radial_shell(test_li, radius=100)) == len(u_real.atoms)


@pytest.mark.parametrize(
    "center_ix, expected_ix",
    [
        (10, [4, 9, 10, 11, 16, 46]),
        (16, [10, 15, 16, 17, 22, 52]),
        (42, [6, 36, 42, 43, 48, 78]),
    ],
)
def test_get_radial_shell_correct_ix_grid(center_ix, expected_ix, u_grid_1):
    # test that the correct ix are being returned, on grid system
    test_atom = u_grid_1.atoms[center_ix]
    shell_ix = get_radial_shell(test_atom, 1.05).resindices
    np.testing.assert_array_equal(shell_ix, expected_ix)
