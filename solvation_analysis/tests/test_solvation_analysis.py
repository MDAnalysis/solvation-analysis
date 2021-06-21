"""
Unit and regression test for the solvation_analysis package.
"""


# Import package, test suite, and other packages as needed
import MDAnalysis as mda
import numpy as np
import pytest

from MDAnalysis.topology.tables import masses
from solvation_analysis.tests.datafiles import (
    bn_fec_data,
    bn_fec_dcd_wrap,
    bn_fec_dcd_unwrap,
    bn_fec_atom_types,
)

from solvation_analysis.solvation import (
    get_atom_group,
    get_closest_n_mol,
    get_radial_shell,
)


def make_grid_universe(n_grid, residue_size, n_frames=10):
    """
    Will make a universe of atoms with specified attributes.

    Parameters
    ----------
    n_grid - dimension of grid sides
    residue_size - size of mda residues
    n_frames - number of frames

    Returns
    -------
    A constructed MDanalysis.Universe
    """
    n_particles = n_grid ** 3
    assert (
        n_particles % residue_size == 0
    ), "residue_size must be a factor of n_particles"
    n_residues = n_particles // residue_size
    atom_residues = np.array(range(0, n_particles)) // residue_size

    grid = np.mgrid[0:n_grid, 0:n_grid, 0:n_grid]  # make the grid
    frame = grid.reshape([3, n_particles]).T  # make it the right shape

    traj = np.empty([n_frames, n_particles, 3])
    for i in range(n_frames):
        traj[i, :, :] = frame  # copy the coordinates to n frames
    u = mda.Universe.empty(
        n_particles, n_residues=n_residues, atom_resindex=atom_residues, trajectory=True
    )  # jam it into a universe
    u.load_new(traj)
    u.add_TopologyAttr("masses", np.ones(n_particles))
    u.add_TopologyAttr("resid")
    return u


@pytest.fixture
def u_grid_3():
    return make_grid_universe(6, 3)


@pytest.fixture
def u_grid_1():
    return make_grid_universe(6, 1)


@pytest.fixture
def u_real():
    return mda.Universe(bn_fec_data, bn_fec_dcd_wrap)


@pytest.fixture
def u_real_named(u_real):
    types = np.loadtxt(bn_fec_atom_types, dtype=str)
    u_real.add_TopologyAttr("name", values=types)
    resnames = ["BN"] * 363 + ["FEC"] * 237 + ["PF6"] * 49 + ["Li"] * 49
    u_real.add_TopologyAttr("resnames", values=resnames)
    return u_real


@pytest.fixture
def atom_groups(u_real):
    li_atoms = u_real.atoms.select_atoms("type 22")
    pf6_atoms = u_real.atoms.select_atoms("byres type 20")
    bn_atoms = u_real.atoms.select_atoms("byres type 5")
    fec_atoms = u_real.atoms.select_atoms("byres type 21")
    atom_groups = {"li": li_atoms, "pf6": pf6_atoms, "bn": bn_atoms, "fec": fec_atoms}
    return atom_groups


def test_get_atom_group(u_real_named):
    u = u_real_named
    res_group = u.residues[1:5]
    atom_group = u.atoms[1:5]
    res = u.residues[1]
    atom = u.atoms[1]
    groups = [res_group, atom_group, res, atom]
    for group in groups:
        assert isinstance(get_atom_group(u, group), mda.core.groups.AtomGroup)


def test_get_closest_n_mol_grid(u_grid_1):
    u = u_grid_1
    test_atoms = u.atoms[[0, 5, 10, 16, 42]]
    for test_atom in test_atoms:
        atoms = get_closest_n_mol(u, test_atom)
        assert len(atoms) == 6


def test_get_closest_n_mol_real(u_real, atom_groups):
    test_li = atom_groups["li"][0]
    shell_sizes = range(2, 8)
    for size in shell_sizes:
        shell = get_closest_n_mol(u_real, test_li, n_mol=size)
        assert len(shell.residues) == size + 1

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


def test_get_radial_shell_grid(u_grid_1):
    u = u_grid_1
    test_atoms = u.atoms[[0, 3, 10, 44]]  # corner, edge, side, center
    distances = [0.95, 1.05, 1.45]
    expected_shell_sizes = {
        0.95: [1, 1, 1, 1],
        1.05: [4, 5, 6, 7],
        1.45: [7, 10, 14, 19],
    }
    for distance in distances:
        expected_sizes = expected_shell_sizes[distance]
        shell_sizes = [len(get_radial_shell(u, atom, distance)) for atom in test_atoms]
        np.testing.assert_allclose(expected_sizes, shell_sizes)


def test_get_radial_shell_real(u_real, atom_groups):
    test_li = atom_groups["li"][0]
    radii_range = range(2, 8)
    shell_sizes = [1, 59, 81, 123, 142, 191]
    for rad, size in zip(radii_range, shell_sizes):
        assert size == len(get_radial_shell(u_real, test_li, radius=rad))

    assert len(get_radial_shell(u_real, test_li, radius=100)) == len(u_real.atoms)


def test_identify_rdf_minimum():
    return
