"""
=========
Selection
=========
:Author: Orion Cohen
:Year: 2021
:Copyright: GNU Public License v3

The selection functions provide a quick and convenient way of selecting the residues
surrounding a specific central species. They all return AtomGroups, useful for
visualization or further analysis.

These functions also serve as methods of the Solution class, selecting the atoms
surrounding specific solutes.
"""

import warnings
import numpy as np
import MDAnalysis as mda
from MDAnalysis.analysis import distances


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
