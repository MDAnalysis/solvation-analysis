"""
Unit and regression test for the solvation_analysis package.
"""

# Import package, test suite, and other packages as needed
import solvation_analysis
import MDAnalysis as mda
import pytest
import sys
import os
import pathlib
import numpy as np
from solvation_analysis.tests.datafiles import bn_fec_data, bn_fec_dcd


def make_grid_universe(n_grid, residue_size, n_frames=10):
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
        n_particles, n_residues=n_residues, atom_resindex=atom_residues
    )  # jam it into a universe
    return u


@pytest.fixture
def u_grid_3(hi):
    return make_grid_universe(6, 3)


@pytest.fixture
def u_grid_1():
    return make_grid_universe(6, 1)


@pytest.fixture
def u_real():
    return mda.Universe(bn_fec_data, bn_fec_dcd)


def test_solvation_analysis_imported():
    """Sample test, will always pass so long as import statement worked"""
    assert "solvation_analysis" in sys.modules


def test_get_atom_groupÎ(u_real):
    print(1)
    print('hello')


def test_get_closest_n_mol():
    return


def test_get_radial_shell():
    return


def test_identify_rdf_minimum():
    return
