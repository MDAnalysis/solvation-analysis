"""
================
Pairing
================
:Author: Orion Cohen, Tingzheng Hou, Kara Fong
:Year: 2021
:Copyright: GNU Public License v3

Elucidate the composition of the the uncoordinated solvent molecules.

Pairing tracks the percent of all solvent molecules paired with the solute, as well
as the composition of the diluent.

While ``pairing`` can be used in isolation, it is meant to be used
as an attribute of the Solution class. This makes instantiating it and calculating the
solvation data a non-issue.
"""

import pandas as pd
import numpy as np

from solvation_analysis._column_names import *



class Pairing:
    """
    Calculate the percent of solutes that are coordinated with each solvent.

    The pairing percentage is the percent of solutes that are coordinated with
    ANY solvent with matching type. So if the pairing of mol1 is 0.5, then 50% of
    solutes are coordinated with at least 1 mol1.

    The pairing percentages are made available as an mean over the whole
    simulation and by frame.

    Parameters
    ----------
    solvation_data : pandas.DataFrame
        The solvation data frame output by Solution.
    n_frames : int
        The number of frames in solvation_data.
    n_solutes : int
        The number of solutes in solvation_data.
    n_solvents : dict of {str: int}
        The number of each kind of solvent.

    Attributes
    ----------
    pairing_dict : dict of {str: float}
        a dictionary where keys are residue names (str) and values are the
        percentage of solutes that contain that residue (float).
    pairing_by_frame : pd.DataFrame
        a dictionary tracking the mean percentage of each residue across frames.
    percent_free_solvents : dict of {str: float}
        a dictionary containing the percent of each solvent that is free. e.g.
        not coordinated to a solute.
    diluent_dict : dict of {str: float}
        the fraction of the diluent constituted by each solvent. The diluent is
        defined as everything that is not coordinated with the solute.
    diluent_by_frame : pd.DataFrame
        a DataFrame of the diluent composition in each frame of the trajectory.
    diluent_counts : pd.DataFrame
        a DataFrame of the raw solvent counts in the diluent in each frame of the trajectory.


    Examples
    --------

     .. code-block:: python

        # first define Li, BN, and FEC AtomGroups
        >>> solution = Solution(Li, {'BN': BN, 'FEC': FEC, 'PF6': PF6})
        >>> solution.run()
        >>> solution.pairing.pairing_dict
        {'BN': 1.0, 'FEC': 0.210, 'PF6': 0.120}
    """

    def __init__(self, solvation_data, n_frames, n_solutes, n_solvents):
        self.solvation_data = solvation_data
        self.n_frames = n_frames
        self.n_solutes = n_solutes
        self.solvent_counts = n_solvents
        self.pairing_dict, self.pairing_by_frame = self._percent_coordinated()
        self.percent_free_solvents = self._percent_free_solvent()
        self.diluent_dict, self.diluent_by_frame, self.diluent_counts = self._diluent_composition()

    @staticmethod
    def from_solution(solution):
        """
        Generate a Pairing object from a solution.

        Parameters
        ----------
        solution : Solution

        Returns
        -------
        Pairing
        """
        assert solution.has_run, "The solution must be run before calling from_solution"
        return Pairing(
            solution.solvation_data,
            solution.n_frames,
            solution.n_solute,
            solution.solvent_counts
        )

    def _percent_coordinated(self):
        # calculate the percent of solute coordinated with each solvent
        counts = self.solvation_data.groupby([FRAME, SOLVATED_ATOM, RESNAME]).count()[RES_IX]
        pairing_series = counts.astype(bool).groupby([RESNAME, FRAME]).sum() / (
            self.n_solutes
        )  # mean coordinated overall
        pairing_by_frame = pairing_series.unstack()
        pairing_normalized = pairing_series / self.n_frames
        pairing_dict = pairing_normalized.groupby([RESNAME]).sum().to_dict()
        return pairing_dict, pairing_by_frame

    def _percent_free_solvent(self):
        # calculate the percent of each solvent NOT coordinated with the solute
        counts = self.solvation_data.groupby([FRAME, RES_IX, RESNAME]).count()[DISTANCE]
        totals = counts.groupby([RESNAME]).count() / self.n_frames
        n_solvents = np.array([self.solvent_counts[name] for name in totals.index.values])
        free_solvents = np.ones(len(totals)) - totals / n_solvents
        return free_solvents.to_dict()

    def _diluent_composition(self):
        coordinated_solvents = self.solvation_data.groupby([FRAME, RESNAME]).nunique()[RES_IX]
        solvent_counts = pd.Series(self.solvent_counts)
        total_solvents = solvent_counts.reindex(coordinated_solvents.index, level=1)
        diluent_solvents = total_solvents - coordinated_solvents
        diluent_series = diluent_solvents / diluent_solvents.groupby([FRAME]).sum()
        diluent_by_frame = diluent_series.unstack().T
        diluent_counts = diluent_solvents.unstack().T
        diluent_dict = diluent_by_frame.mean(axis=1).to_dict()
        return diluent_dict, diluent_by_frame, diluent_counts

