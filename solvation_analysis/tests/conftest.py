import sys
import MDAnalysis as mda
import numpy as np
import pandas as pd
import pytest

from MDAnalysis import transformations
from solvation_analysis.networking import Networking
from solvation_analysis.residence import Residence

from solvation_analysis.rdf_parser import identify_cutoff_poly
from solvation_analysis.tests.datafiles import (
    bn_fec_data,
    bn_fec_dcd_wrap,
    bn_fec_dcd_unwrap,
    bn_fec_atom_types,
    eax_data,
    iba_data,
    iba_dcd,
)
from solvation_analysis.tests.datafiles import (
    easy_rdf_bins,
    easy_rdf_data,
    hard_rdf_bins,
    hard_rdf_data,
    non_solv_rdf_bins,
    non_solv_rdf_data,
    bn_fec_solv_df_large,
)
from solvation_analysis.solute import Solute
import pathlib


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


@pytest.fixture(scope='module')
def u_real():
    """Returns a universe of a BN FEC trajectory"""
    return mda.Universe(bn_fec_data, bn_fec_dcd_wrap)


@pytest.fixture(scope='module')
def u_real_named(u_real):
    """Returns a universe of a BN FEC trajectory with residues and atoms named"""
    types = np.loadtxt(bn_fec_atom_types, dtype=str)
    u_real.add_TopologyAttr("name", values=types)
    resnames = ["BN"] * 363 + ["FEC"] * 237 + ["PF6"] * 49 + ["Li"] * 49
    u_real.add_TopologyAttr("resnames", values=resnames)
    return u_real


@pytest.fixture(scope='module')
def atom_groups(u_real):
    """Returns pre-selected atom groups in the BN FEC universe"""
    li_atoms = u_real.atoms.select_atoms("type 22")
    pf6_atoms = u_real.atoms.select_atoms("byres type 20")
    bn_atoms = u_real.atoms.select_atoms("byres type 5")
    fec_atoms = u_real.atoms.select_atoms("byres type 19")
    atom_groups = {"li": li_atoms, "pf6": pf6_atoms, "bn": bn_atoms, "fec": fec_atoms}
    return atom_groups


def rdf_loading_helper(bins_files, data_files):
    """
    Creates dictionary of bin and data arrays with a rdf tag as key
    """
    rdf_bins = {
        key: list(np.load(npz).values())[0] for key, npz in bins_files.items()
    }
    rdf_data = {
        key: list(np.load(npz).values())[0] for key, npz in data_files.items()
    }
    shared_keys = set(rdf_data.keys()) & set(rdf_bins.keys())
    rdf_bins_and_data = {key: (rdf_bins[key], rdf_data[key]) for key in shared_keys}
    return rdf_bins_and_data


@pytest.fixture
def rdf_bins_and_data_easy():
    return rdf_loading_helper(easy_rdf_bins, easy_rdf_data)


@pytest.fixture
def rdf_bins_and_data_hard():
    return rdf_loading_helper(hard_rdf_bins, hard_rdf_data)


@pytest.fixture
def rdf_bins_and_data_non_solv():
    return rdf_loading_helper(non_solv_rdf_bins, non_solv_rdf_data)


@pytest.fixture(scope='module')
def pre_solute(atom_groups):
    li = atom_groups['li']
    pf6 = atom_groups['pf6']
    bn = atom_groups['bn']
    fec = atom_groups['fec']
    return Solute.from_atoms(
        li,
        {'pf6': pf6, 'bn': bn, 'fec': fec},
        radii={'pf6': 2.8, 'bn': 2.61468, 'fec': 2.43158},
        rdf_init_kwargs={"range": (0, 8.0)},
    )


@pytest.fixture(scope='function')
def pre_solute_mutable(atom_groups):
    li = atom_groups['li']
    pf6 = atom_groups['pf6']
    bn = atom_groups['bn']
    fec = atom_groups['fec']
    return Solute.from_atoms(
        li,
        {'pf6': pf6, 'bn': bn, 'fec': fec},
        radii={'pf6': 2.8, 'bn': 2.61468, 'fec': 2.43158},
        rdf_init_kwargs={"range": (0, 8.0)},
        rdf_kernel=identify_cutoff_poly,
    )


@pytest.fixture(scope='module')
def run_solute(pre_solute):
    pre_solute.run(step=1)
    return pre_solute


@pytest.fixture(scope='module')
def u_eax_series():
    boxes = {
        'ea': [45.760393, 45.760393, 45.760393, 90, 90, 90],
        'eaf': [47.844380, 47.844380, 47.844380, 90, 90, 90],
        'fea': [48.358954, 48.358954, 48.358954, 90, 90, 90],
        'feaf': [50.023129, 50.023129, 50.023129, 90, 90, 90],
    }
    us = {}
    for solvent_dir in pathlib.Path(eax_data).iterdir():
        u_solv = mda.Universe(
            str(solvent_dir / 'topology.pdb'),
            str(solvent_dir / 'trajectory_equil.dcd')
        )
        # our dcd lacks dimensions so we must manually set them
        box = boxes[solvent_dir.stem]
        set_dim = transformations.boxdimensions.set_dimensions(box)
        u_solv.trajectory.add_transformations(set_dim)
        us[solvent_dir.stem] = u_solv
    return us


@pytest.fixture(scope='module')
def u_eax_atom_groups(u_eax_series):
    atom_groups_dict = {}
    for name, u in u_eax_series.items():
        atom_groups = {}
        atom_groups['li'] = u.atoms.select_atoms("element Li")
        atom_groups['pf6'] = u.atoms.select_atoms("byres element P")
        residue_lengths = np.array([len(elements) for elements in u.residues.elements])
        eax_fec_cutoff = np.unique(residue_lengths, return_index=True)[1][2]
        atom_groups[name] = u.atoms.select_atoms(f"resid 1:{eax_fec_cutoff}")
        atom_groups['fec'] = u.atoms.select_atoms(f"resid {eax_fec_cutoff + 1}:600")
        atom_groups_dict[name] = atom_groups
    return atom_groups_dict


@pytest.fixture(scope='module')
def eax_solutes(u_eax_atom_groups):
    solutes = {}
    for name, atom_groups in u_eax_atom_groups.items():
        solute = Solute.from_atoms(
            atom_groups['li'],
            {'pf6': atom_groups['pf6'], name: atom_groups[name], 'fec': atom_groups['fec']},
        )
        solute.run()
        solutes[name] = solute
    return solutes

@pytest.fixture(scope='module')
def iba_u():
    return mda.Universe(iba_data, iba_dcd)

@pytest.fixture(scope='module')
def iba_solvents(iba_u):
    iba = iba_u.select_atoms("byres element C")
    H2O = iba_u.atoms - iba
    return {'iba': iba, 'H2O': H2O}

@pytest.fixture(scope='module')
def iba_atom_groups(iba_solvents):
    iba = iba_solvents['iba']
    return {
        'iba_alcohol_O': iba[5::12],
        'iba_alcohol_H': iba[11::12],
        'iba_ketone': iba[4::12],
        'iba_C0': iba[0::12],
        'iba_C1': iba[1::12],
        'iba_C2': iba[2::12],
        'iba_C3': iba[3::12],
        'iba_H6': iba[6::12],
        'iba_H7': iba[7::12],
        'iba_H8': iba[8::12],
        'iba_H9': iba[9::12],
        'iba_H10': iba[10::12],
    }


@pytest.fixture(scope='module')
def H2O_atom_groups(iba_solvents):
    H2O = iba_solvents['H2O']
    return {
        'H2O_O': H2O[0::3],
        'H2O_H1': H2O[1::3],
        'H2O_H2': H2O[2::3],
    }


@pytest.fixture(scope='module')
def iba_solutes(iba_atom_groups, iba_solvents):
    solutes = {}
    for name, atom_group in iba_atom_groups.items():
        radii = {'iba': 1.9, 'H2O': 1.9} if ('iba_H' in name or 'iba_C' in name) else None
        solute = Solute.from_atoms(
            atom_group,
            iba_solvents,
            solute_name=name,
            radii=radii,
        )
        solute.run()
        solutes[name] = solute
    return solutes


@pytest.fixture(scope='module')
def iba_small_solute(iba_atom_groups, iba_solvents):
    solute_atoms = {
        'iba_ketone': iba_atom_groups['iba_ketone'],
        'iba_alcohol_O': iba_atom_groups['iba_alcohol_O'],
        'iba_alcohol_H': iba_atom_groups['iba_alcohol_H']
    }
    solute = Solute.from_atoms_dict(
        solute_atoms,
        iba_solvents,
        solute_name='iba',
    )
    solute.run()
    return solute


@pytest.fixture
def solvation_results(run_solute):
    return run_solute._solvation_frames


@pytest.fixture
def solvation_data(run_solute):
    return run_solute.solvation_data


@pytest.fixture(scope='module')
def solvation_data_large():
    return pd.read_csv(bn_fec_solv_df_large, index_col=[0, 1, 2, 3])


@pytest.fixture(scope='module')
def solvation_data_sparse(solvation_data_large):
    step = 10
    return solvation_data_large.loc[pd.IndexSlice[::step, :, :], :]


@pytest.fixture(scope='module')
def residence(solvation_data_sparse):
    return Residence(solvation_data_sparse, step=10)


@pytest.fixture(scope='module')
def networking(run_solute):
    return Networking.from_solute(run_solute, 'pf6')