import sys
import MDAnalysis as mda
import numpy as np
import pytest

from solvation_analysis.tests.datafiles import (
    bn_fec_data,
    bn_fec_dcd_wrap,
    bn_fec_dcd_unwrap,
    bn_fec_atom_types,
)

from solvation_analysis.tests.datafiles import (
    rdf_bn_all_bins,
    rdf_bn_all_data,
    rdf_bn_N_bins,
    rdf_bn_N_data,
    rdf_fec_all_bins,
    rdf_fec_all_data,
    rdf_fec_F_bins,
    rdf_fec_F_data,
    rdf_fec_O_bins,
    rdf_fec_O_data,
    rdf_pf6_all_bins,
    rdf_pf6_all_data,
    rdf_pf6_F_bins,
    rdf_pf6_F_data,
    rdf_universe_all_bins,
    rdf_universe_all_data,
)


def test_solvation_analysis_imported():
    """Sample test, will always pass so long as import statement worked"""
    assert "solvation_analysis" in sys.modules


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
    """Creates a 6x6x6 grid with residues containing 3 atoms"""
    return make_grid_universe(6, 3)


@pytest.fixture
def u_grid_1():
    """Creates a 6x6x6 grid with residues containing 1 atom"""
    return make_grid_universe(6, 1)


@pytest.fixture
def u_real():
    """Returns a universe of a BN FEC trajectory"""
    return mda.Universe(bn_fec_data, bn_fec_dcd_wrap)


@pytest.fixture
def u_real_named(u_real):
    """Returns a universe of a BN FEC trajectory with residues and atoms named"""
    types = np.loadtxt(bn_fec_atom_types, dtype=str)
    u_real.add_TopologyAttr("name", values=types)
    resnames = ["BN"] * 363 + ["FEC"] * 237 + ["PF6"] * 49 + ["Li"] * 49
    u_real.add_TopologyAttr("resnames", values=resnames)
    return u_real


@pytest.fixture
def atom_groups(u_real):
    """Returns pre-selected atom groups in the BN FEC universe"""
    li_atoms = u_real.atoms.select_atoms("type 22")
    pf6_atoms = u_real.atoms.select_atoms("byres type 20")
    bn_atoms = u_real.atoms.select_atoms("byres type 5")
    fec_atoms = u_real.atoms.select_atoms("byres type 21")
    atom_groups = {"li": li_atoms, "pf6": pf6_atoms, "bn": bn_atoms, "fec": fec_atoms}
    return atom_groups


@pytest.fixture
def rdf_bins():
    rdf_bin_files = {
        "bn_all": rdf_bn_all_bins,
        "bn_N": rdf_bn_N_bins,
        "fec_all": rdf_fec_all_bins,
        "fec_F": rdf_fec_F_bins,
        "fec_O": rdf_fec_O_bins,
        "pf6_all": rdf_pf6_all_bins,
        "pf6_F": rdf_pf6_F_bins,
        "universe_all": rdf_universe_all_bins,
    }
    rdf_bins = {
        key: np.genfromtxt(csv, delimiter=", ") for key, csv in rdf_bin_files.items()
    }
    return rdf_bins


@pytest.fixture
def rdf_data():
    rdf_data_files = {
        "bn_all": rdf_bn_all_data,
        "bn_N": rdf_bn_N_data,
        "fec_all": rdf_fec_all_data,
        "fec_F": rdf_fec_F_data,
        "fec_O": rdf_fec_O_data,
        "pf6_all": rdf_pf6_all_data,
        "pf6_F": rdf_pf6_F_data,
        "universe_all": rdf_universe_all_data,
    }
    rdf_data = {
        key: np.genfromtxt(csv, delimiter=", ") for key, csv in rdf_data_files.items()
    }
    return rdf_data
