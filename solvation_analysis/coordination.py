"""
================
Coordination
================
:Author: Orion Cohen, Tingzheng Hou, Kara Fong
:Year: 2021
:Copyright: GNU Public License v3

Elucidate the coordination of each solvating species.

Coordination numbers for each solvent are automatically calculated, along with
the types of every coordinating atom.

While ``coordination`` can be used in isolation, it is meant to be used
as an attribute of the Solute class. This makes instantiating it and calculating the
solvation data a non-issue.
"""

import pandas as pd

from solvation_analysis._column_names import *


class Coordination:
    """
    Calculate the coordination number for each solvent.

    Coordination calculates the coordination number by averaging the number of
    coordinated solvents in all of the solvation shells. This is equivalent to
    the typical method of integrating the solute-solvent RDF up to the solvation
    radius cutoff. As a result, Coordination calculates **species-species** coordination
    numbers, not the total coordination number of the solute. So if the coordination
    number of mol1 is 3.2, there are on average 3.2 mol1 residues within the solvation
    distance of each solute.

    The coordination numbers are made available as an mean over the whole
    simulation and by frame.

    Parameters
    ----------
    solvation_data : pandas.DataFrame
        The solvation data frame output by Solute.
    n_frames : int
        The number of frames in solvation_data.
    n_solutes : int
        The number of solutes in solvation_data.

    Attributes
    ----------
    cn_dict : dict of {str: float}
        a dictionary where keys are residue names (str) and values are the
        mean coordination number of that residue (float).
    cn_by_frame : pd.DataFrame
        a DataFrame of the mean coordination number of in each frame of the trajectory.
    coordinating_atoms : pd.DataFrame
        fraction of each atom_type participating in solvation, calculated for each solvent.

    Examples
    --------

     .. code-block:: python

        # first define Li, BN, and FEC AtomGroups
        >>> solute = Solute(Li, {'BN': BN, 'FEC': FEC, 'PF6': PF6})
        >>> solute.run()
        >>> solute.coordination.cn_dict
        {'BN': 4.328, 'FEC': 0.253, 'PF6': 0.128}

    """

    def __init__(self, solvation_data, n_frames, n_solutes, atom_group):
        self.solvation_data = solvation_data
        self.n_frames = n_frames
        self.n_solutes = n_solutes
        self.cn_dict, self.cn_by_frame = self._mean_cn()
        self.atom_group = atom_group
        self.coordinating_atoms = self._calculate_coordinating_atoms()

    @staticmethod
    def from_solute(solute):
        """
        Generate a Coordination object from a solute.

        Parameters
        ----------
        solute : Solute

        Returns
        -------
        Pairing
        """
        assert solute.has_run, "The solute must be run before calling from_solute"
        return Coordination(
            solute.solvation_data,
            solute.n_frames,
            solute.n_solutes,
            solute.u.atoms,
        )

    def _mean_cn(self):
        counts = self.solvation_data.groupby([FRAME, SOLUTE_IX, SOLVENT]).count()[SOLVENT_IX]
        cn_series = counts.groupby([SOLVENT, FRAME]).sum() / (
                self.n_solutes * self.n_frames
        )
        cn_by_frame = cn_series.unstack()
        cn_dict = cn_series.groupby([SOLVENT]).sum().to_dict()
        return cn_dict, cn_by_frame

    def _calculate_coordinating_atoms(self, tol=0.005):
        """
        Determine which atom types are actually coordinating
        return the types of those atoms
        """
        # lookup atom types
        atom_types = self.solvation_data.reset_index([SOLVENT_ATOM_IX])
        atom_types[ATOM_TYPE] = self.atom_group[atom_types[SOLVENT_ATOM_IX].values].types
        # count atom types
        atoms_by_type = atom_types[[ATOM_TYPE, SOLVENT, SOLVENT_ATOM_IX]]
        type_counts = atoms_by_type.groupby([SOLVENT, ATOM_TYPE]).count()
        solvent_counts = type_counts.groupby([SOLVENT]).sum()[SOLVENT_ATOM_IX]
        # calculate fraction of each
        solvent_counts_list = [
            solvent_counts[solvent] for solvent in
            type_counts.index.get_level_values(SOLVENT)
        ]
        type_fractions = type_counts[SOLVENT_ATOM_IX] / solvent_counts_list
        type_fractions.name = FRACTION
        # change index type
        type_fractions = (type_fractions
                         .reset_index(ATOM_TYPE)
                         .astype({ATOM_TYPE: str})
                         .set_index(ATOM_TYPE, append=True)
                         )
        return type_fractions[type_fractions[FRACTION] > tol]
