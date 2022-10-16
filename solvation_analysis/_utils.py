import numpy as np
from collections import defaultdict
from functools import reduce
import MDAnalysis as mda
from MDAnalysis.analysis import distances

from solvation_analysis._column_names import *


def verify_solute_atoms(solute_atom_group):
    # we presume that the solute_atoms has the same number of atoms on each residue
    # and that they all have the same indices on those residues
    # and that the residues are all the same length
    # then this should work
    all_res_len = np.array([res.atoms.n_atoms for res in solute_atom_group.residues])
    assert np.all(all_res_len[0] == all_res_len), (
        "All residues must be the same length."
    )
    res_atom_local_ix = defaultdict(list)
    res_atom_ix = defaultdict(list)

    for atom in solute_atom_group.atoms:
        res_atom_local_ix[atom.resindex].append(atom.ix - atom.residue.atoms[0].ix)
        res_atom_ix[atom.resindex].append(atom.index)
    res_occupancy = np.array([len(ix) for ix in res_atom_local_ix.values()])
    assert np.all(res_occupancy[0] == res_occupancy), (
        "All residues must have the same number of solute_atoms atoms on them."
    )

    res_atom_array = np.array(list(res_atom_local_ix.values()))
    assert np.all(res_atom_array[0] == res_atom_array), (
        "All residues must have the same solute_atoms atoms on them."
    )

    res_atom_ix_array = np.array(list(res_atom_ix.values()))
    solute_atom_group_dict = {}
    for i in range(0, res_atom_ix_array.shape[1]):
        solute_atom_group_dict[i] = solute_atom_group.universe.atoms[res_atom_ix_array[:, i]]
    return solute_atom_group_dict


def verify_solute_atoms_dict(solute_atoms_dict):
    # first we verify the input format
    atom_group_lengths = []
    for solute_name, solute_atom_group in solute_atoms_dict.items():
        assert isinstance(solute_name, str), (
            "The keys of solutes_dict must be strings."
        )
        assert isinstance(solute_atom_group, mda.AtomGroup), (
            f"The values of solutes_dict must be MDAnalysis.AtomGroups. But the value"
            f"for {solute_name} is a {type(solute_atom_group)}."
        )
        assert len(solute_atom_group) == len(solute_atom_group.residues), (
            "The solute_atom_group must have a single atom on each residue."
        )
        atom_group_lengths.append(len(solute_atom_group))
    assert np.all(np.array(atom_group_lengths) == atom_group_lengths[0]), (
        "AtomGroups in solutes_dict must have the same length because there should be"
        "one atom per solute residue."
    )

    # verify that the solute_atom_groups have no overlap
    solute_atom_group = reduce(lambda x, y: x | y, [atoms for atoms in solute_atoms_dict.values()])
    assert solute_atom_group.n_atoms == sum([atoms.n_atoms for atoms in solute_atoms_dict.values()]), (
        "The solute_atom_groups must not overlap."
    )
    verify_solute_atoms(solute_atom_group)

    return solute_atom_group


def get_atom_group(selection):
    """
    Cast an MDAnalysis.Atom, MDAnalysis.Residue, or MDAnalysis.ResidueGroup to AtomGroup.

    Parameters
    ----------
    selection: MDAnalysis.Atom, MDAnalysis.Residue or MDAnalysis.ResidueGroup
        atoms to cast

    Returns
    -------
    MDAnalysis.AtomGroup

    """
    assert isinstance(
        selection,
        (
            mda.core.groups.Residue,
            mda.core.groups.ResidueGroup,
            mda.core.groups.Atom,
            mda.core.groups.AtomGroup,
        ),
    ), "central_species must be one of the preceding types"
    u = selection.universe
    if isinstance(selection, (mda.core.groups.Residue, mda.core.groups.ResidueGroup)):
        selection = selection.atoms
    if isinstance(selection, mda.core.groups.Atom):
        selection = u.select_atoms(f"index {selection.index}")
    return selection


def get_closest_n_mol(
    central_species,
    n_mol,
    guess_radius=3,
    return_ordered_resix=False,
    return_radii=False,
):
    """
    Returns the closest n molecules to the central species, an array of their resix,
    and an array of the distance of the closest atom in each molecule.

    Parameters
    ----------
    central_species : MDAnalysis.Atom, MDAnalysis.AtomGroup, MDAnalysis.Residue or MDAnalysis.ResidueGroup
    n_mol : int
        The number of molecules to return
    guess_radius : float or int
        an initial search radius to look for closest n mol
    return_ordered_resix : bool, default False
        whether to return the resix of the closest n
        molecules, ordered by radius
    return_radii : bool, default False
        whether to return the distance of the closest atom of each
        of the n molecules

    Returns
    -------
    full shell : MDAnalysis.AtomGroup
        the atoms in the shell
    ordered_resix : numpy.array of int
        the residue index of the n_mol closest atoms
    radii : numpy.array of float
        the distance of each atom from the center
    """
    u = central_species.universe
    central_species = get_atom_group(central_species)
    coords = central_species.center_of_mass()
    pairs, radii = mda.lib.distances.capped_distance(
        coords, u.atoms.positions, guess_radius, return_distances=True, box=u.dimensions
    )
    partial_shell = u.atoms[pairs[:, 1]]
    shell_resix = partial_shell.resindices
    if len(np.unique(shell_resix)) < n_mol + 1:
        return get_closest_n_mol(
            central_species,
            n_mol,
            guess_radius + 1,
            return_ordered_resix=return_ordered_resix,
            return_radii=return_radii
        )
    radii = distances.distance_array(coords, partial_shell.positions, box=u.dimensions)[
        0
    ]
    ordering = np.argsort(radii)
    ordered_resix = shell_resix[ordering]
    closest_n_resix = np.sort(np.unique(ordered_resix, return_index=True)[1])[
        0: n_mol + 1
    ]
    str_resix = " ".join(str(resix) for resix in ordered_resix[closest_n_resix])
    full_shell = u.select_atoms(f"resindex {str_resix}")
    if return_ordered_resix and return_radii:
        return (
            full_shell,
            ordered_resix[closest_n_resix],
            radii[ordering][closest_n_resix],
        )
    elif return_ordered_resix:
        return full_shell, ordered_resix[closest_n_resix]
    elif return_radii:
        return full_shell, radii[ordering][closest_n_resix]
    else:
        return full_shell


def get_radial_shell(central_species, radius):
    """
    Returns all molecules with atoms within the radius of the central species.
    (specifically, within the radius of the COM of central species).

    Parameters
    ----------
    central_species : MDAnalysis.Atom, MDAnalysis.AtomGroup, MDAnalysis.Residue, or MDAnalysis.ResidueGroup
    radius : float or int
        radius used for atom selection

    Returns
    -------
    MDAnalysis.AtomGroup

    """
    u = central_species.universe
    central_species = get_atom_group(central_species)
    coords = central_species.center_of_mass()
    str_coords = " ".join(str(coord) for coord in coords)
    partial_shell = u.select_atoms(f"point {str_coords} {radius}")
    full_shell = partial_shell.residues.atoms
    return full_shell


def calculate_adjacency_dataframe(solvation_data):
    """
    Calculate a frame-by-frame adjacency matrix from the solvation data.

    This will calculate the adjacency matrix of the solute and all possible
    solvents. It will maintain an index of ["frame", "solute_atom", "solvent"]
    where each "frame" is a sparse adjacency matrix between solvated atom ix
    and residue ix.

    Parameters
    ----------
    solvation_data : pd.DataFrame
        the solvation_data from a Solute.

    Returns
    -------
    adjacency_df : pandas.DataFrame
    """
    # generate an adjacency matrix from the solvation data
    adjacency_group = solvation_data.groupby([FRAME, SOLUTE_IX, SOLVENT_IX])
    adjacency_df = adjacency_group[DISTANCE].count().unstack(fill_value=0)
    return adjacency_df
