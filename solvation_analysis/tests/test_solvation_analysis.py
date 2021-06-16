"""
Unit and regression test for the solvation_analysis package.
"""

import sys

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
        traj[i, :, :] = frame  # copy the coordinates to 10 frames
    u = mda.Universe.empty(
        n_particles, n_residues=n_residues, atom_resindex=atom_residues, trajectory=True
    )  # jam it into a universe
    u.add_TopologyAttr("masses", np.ones(n_particles))
    u.add_TopologyAttr("resid")
    return u


@pytest.fixture
def u_grid_3(hi):
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
    pf6_atoms = u_real.atoms.select_atoms("type 20").residues.atoms
    bn_atoms = u_real.atoms.select_atoms("type 5").residues.atoms
    fec_atoms = u_real.atoms.select_atoms("type 21").residues.atoms
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


def test_get_closest_n_mol(u_grid_1, u_real, atom_groups):
    # TODO: fix grid trajectory
    # TODO: use grid trajectory to test atom closeness / res picking
    # u = u_grid_1
    # test_atom = u.atoms[42]
    # atoms = get_closest_n_mol(u, test_atom)
    # assert len(atoms) == 5
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
        np.testing.assert_equal(resids, default_resids)
        np.testing.assert_equal(radii, default_radii)


def test_get_radial_shell(u_grid_1, u_real, atom_groups):
    test_li = atom_groups["li"][0]
    radii_range = range(2, 8)
    shell_sizes = [1, 59, 81, 123, 142, 191]
    for rad, size in zip(radii_range, shell_sizes):
        assert size == len(get_radial_shell(u_real, test_li, radius=rad))

    assert len(get_radial_shell(u_real, test_li, radius=100)) == len(u_real.atoms)





def test_identify_rdf_minimum():
    return
