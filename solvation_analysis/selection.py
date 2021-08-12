"""
=========
selection
=========
:Author: Orion Cohen
:Year: 2021
:Copyright: GNU Public License v3

Convenient selection utilities.
"""

import warnings
import numpy as np
import MDAnalysis as mda
from MDAnalysis.analysis import distances
import nglview as nv


def visualize(selection):
    mda_view = nv.show_mdanalysis(selection)
    mda_view.add_representation("licorice", selection="Li", color="blue")
    return mda_view.display()


def get_atom_group(selection):
    """
    Cast an Atom, AtomGroup, Residue, or ResidueGroup to AtomGroup.

    Parameters
    ----------
    selection: Atom, AtomGroup, Residue or ResidueGroup
        atoms to cast

    Returns
    -------
    AtomGroup

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


def _get_n_shells(central_species, n_shell=2, radius=3, ignore_atoms=None):
    """
    CURRENTLY INCOMPLETE, set to private until complete

    A list containing the nth shell at the nth index. Note that the shells
    have 0 intersection. For example, calling get_n_shells with n_shell = 2
    would return: [central_species, first_shell, second_shell]. This scales
    factorially so probably don't go over n_shell = 3

    Parameters
    ----------
    central_species : Atom, AtomGroup, Residue or ResidueGroup
    n_shell : int
        number of shells to return
    radius : float or int
        radius used to select atoms in next shell
    ignore_atoms : AtomGroup
        these atoms will be ignored

    Returns
    -------
    List of AtomGroups


    """
    u = central_species.universe
    if n_shell > 3:
        warnings.warn("get_n_shells scales factorially, very slow")
    central_species = get_atom_group(central_species)
    if not ignore_atoms:
        ignore_atoms = u.select_atoms("")


def get_closest_n_mol(
    central_species,
    n_mol,
    guess_radius=3,
    return_ordered_resids=False,
    return_radii=False,
):
    """
    Returns the closest n molecules to the central species, an array of their resids,
    and an array of the distance of the closest atom in each molecule.

    Parameters
    ----------
    central_species : Atom, AtomGroup, Residue or ResidueGroup
    n_mol : int
        The number of molecules to return
    guess_radius : float or int
        an initial search radius to look for closest n mol
    return_ordered_resids : bool, default False
        whether to return the resids of the closest n
        molecules, ordered by radius
    return_radii : bool, default False
        whether to return the distance of the closest atom of each
        of the n molecules

    Returns
    -------
    full shell : AtomGroup
        the atoms in the shell
    ordered_resids : numpy.array of int
        the residue id of the n_mol closest atoms
    radii : numpy.array of float
        the distance of each atom from the center
    """
    u = central_species.universe
    central_species = get_atom_group(central_species)
    coords = central_species.center_of_mass()
    str_coords = " ".join(str(coord) for coord in coords)
    pairs, radii = mda.lib.distances.capped_distance(
        coords, u.atoms.positions, guess_radius, return_distances=True, box=u.dimensions
    )
    partial_shell = u.atoms[pairs[:, 1]]
    shell_resids = partial_shell.resids
    if len(np.unique(shell_resids)) < n_mol + 1:
        return get_closest_n_mol(
            central_species,
            n_mol,
            guess_radius + 1,
            return_ordered_resids=return_ordered_resids,
            return_radii=return_radii
        )
    radii = distances.distance_array(coords, partial_shell.positions, box=u.dimensions)[
        0
    ]
    ordering = np.argsort(radii)
    ordered_resids = shell_resids[ordering]
    closest_n_resix = np.sort(np.unique(ordered_resids, return_index=True)[1])[
        0: n_mol + 1
    ]
    str_resids = " ".join(str(resid) for resid in ordered_resids[closest_n_resix])
    full_shell = u.select_atoms(f"resid {str_resids}")
    if return_ordered_resids and return_radii:
        return (
            full_shell,
            ordered_resids[closest_n_resix],
            radii[ordering][closest_n_resix],
        )
    elif return_ordered_resids:
        return full_shell, ordered_resids[closest_n_resix]
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
    central_species : Atom, AtomGroup, Residue, or ResidueGroup
    radius : float or int
        radius used for atom selection

    Returns
    -------
    AtomGroup

    """
    u = central_species.universe
    central_species = get_atom_group(central_species)
    coords = central_species.center_of_mass()
    str_coords = " ".join(str(coord) for coord in coords)
    partial_shell = u.select_atoms(f"point {str_coords} {radius}")
    full_shell = partial_shell.residues.atoms
    return full_shell
