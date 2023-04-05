"""
================
Speciation
================
:Author: Orion Cohen, Tingzheng Hou, Kara Fong
:Year: 2021
:Copyright: GNU Public License v3

Explore the precise solvation shell of every solute.

Speciation tabulates the unique solvation shell compositions, their fraction,
and their temporal locations.

From this, it provides search functionality to query for specific solvation shell
compositions. Extremely convenient for visualization.

While ``speciation`` can be used in isolation, it is meant to be used
as an attribute of the Solute class. This makes instantiating it and calculating the
solvation data a non-issue.
"""

import pandas as pd

from solvation_analysis._column_names import *


class Speciation:
    """
    Calculate the solvation shells of every solute.

    Speciation organizes the solvation data by the type of residue
    coordinated with the central solvent. It collects this information in a
    pandas.DataFrame indexed by the frame and solute number. Each column is
    one of the solvents in the solvent_name column of the solvation data. The
    column value is how many residue of that type are in the solvation shell.

    Speciation provides the speciation of each solute in the speciation
    attribute, it also calculates the fraction of each unique
    shell and makes it available in the speciation_fraction attribute.

    Additionally, there are methods for finding solvation shells of
    interest and computing how common certain shell configurations are.

    Parameters
    ----------
    solvation_data : pandas.DataFrame
        The solvation data frame output by Solute.
    n_frames : int
        The number of frames in solvation_data.
    n_solutes : int
        The number of solutes in solvation_data.
    """

    def __init__(self, solvation_data, n_frames, n_solutes):
        self.solvation_data = solvation_data
        self.n_frames = n_frames
        self.n_solutes = n_solutes
        self._speciation_df, self._speciation_fraction = self._compute_speciation()
        self._solvent_co_occurrence = self._solvent_co_occurrence()

    @staticmethod
    def from_solute(solute):
        """
        Generate a Speciation object from a solute.

        Parameters
        ----------
        solute : Solute

        Returns
        -------
        Pairing
        """
        assert solute.has_run, "The solute must be run before calling from_solute"
        return Speciation(
            solute.solvation_data,
            solute.n_frames,
            solute.n_solutes,
        )

    def _compute_speciation(self):
        counts = self.solvation_data.groupby([FRAME, SOLUTE_IX, SOLVENT]).count()[SOLVENT_IX]
        counts_re = counts.reset_index([SOLVENT])
        speciation_data = counts_re.pivot(columns=[SOLVENT]).fillna(0).astype(int)
        res_names = speciation_data.columns.levels[1]
        speciation_data.columns = res_names
        sum_series = speciation_data.groupby(speciation_data.columns.to_list()).size()
        sum_sorted = sum_series.sort_values(ascending=False)
        speciation_fraction = sum_sorted.reset_index().rename(columns={0: COUNT})
        speciation_fraction[COUNT] = speciation_fraction[COUNT] / (self.n_frames * self.n_solutes)
        return speciation_data, speciation_fraction

    @classmethod
    def _mean_speciation(cls, speciation_frames, solute_number, frame_number):
        means = speciation_frames.sum(axis=1) / (solute_number * frame_number)
        return means

    def calculate_shell_fraction(self, shell_dict):
        """
        Calculate the fraction of shells matching shell_dict.

        This function computes the fraction of solvation shells that exist with a particular
        composition. The composition is specified by the shell_dict. The fraction
        will be of all shells that match that specification.

        Attributes
        ----------
        shell_dict : dict of {str: int}
            a specification for a shell composition. Keys are residue names (str)
            and values are the number of desired residues. e.g. if shell_dict =
            {'mol1': 4} then the function will return the fraction of shells
            that have 4 mol1. Note that this may include shells with 4 mol1 and
            any number of other solvents. To specify a shell with 4 mol1 and nothing
            else, enter a dict such as {'mol1': 4, 'mol2': 0, 'mol3': 0}.

        Returns
        -------
        float
            the fraction of shells

        Examples
        --------

         .. code-block:: python

            # first define Li, BN, and FEC AtomGroups
            >>> solute = Solute(Li, {'BN': BN, 'FEC': FEC, 'PF6': PF6})
            >>> solute.run()
            >>> solute.speciation.calculate_shell_fraction({'BN': 4, 'PF6': 1})
            0.0898
        """
        query_list = [f"{name} == {str(count)}" for name, count in shell_dict.items()]
        query = " and ".join(query_list)
        query_counts = self.speciation_fraction.query(query)
        return query_counts[COUNT].sum()

    def get_shells(self, shell_dict):
        """
        Find all solvation shells that match shell_dict.

        This returns the frame, solute index, and composition of all solutes
        that match the composition given in shell_dict.

        Attributes
        ----------
        shell_dict : dict of {str: int}
            a specification for a shell composition. Keys are residue names (str)
            and values are the number of desired residues. e.g. if shell_dict =
            {'mol1': 4} then the function will return all shells
            that have 4 mol1. Note that this may include shells with 4 mol1 and
            any number of other solvents. To specify a shell with 4 mol1 and nothing
            else, enter a dict such as {'mol1': 4, 'mol2': 0, 'mol3': 0}.

        Returns
        -------
        pandas.DataFrame
            the index and composition of all shells that match shell_dict
        """
        query_list = [f"{name} == {str(count)}" for name, count in shell_dict.items()]
        query = " and ".join(query_list)
        query_counts = self.speciation_data.query(query)
        return query_counts

    def _solvent_co_occurrence(self):
        # calculate the co-occurrence of solvent molecules.
        expected_solvents_list = []
        actual_solvents_list = []
        for solvent in self.speciation_data.columns.values:
            # calculate number of available coordinating solvent slots
            shells_w_solvent = self.speciation_data.query(f'{solvent} > 0')
            n_solvents = shells_w_solvent.sum()
            # calculate expected number of coordinating solvents
            n_coordination_slots = n_solvents.sum() - len(shells_w_solvent)
            coordination_fraction = self.speciation_data.sum() / self.speciation_data.sum().sum()
            expected_solvents = coordination_fraction * n_coordination_slots
            # calculate actual number of coordinating solvents
            actual_solvents = n_solvents.copy()
            actual_solvents[solvent] = actual_solvents[solvent] - len(shells_w_solvent)
            # name series and append to list
            expected_solvents.name = solvent
            actual_solvents.name = solvent
            expected_solvents_list.append(expected_solvents)
            actual_solvents_list.append(actual_solvents)
        if len(actual_solvents_list) == 0 or len(expected_solvents_list) == 0:
            # we return this if nothing is solvated
            return pd.DataFrame()
        # make DataFrames
        actual_df = pd.concat(actual_solvents_list, axis=1)
        expected_df = pd.concat(expected_solvents_list, axis=1)
        # calculate correlation matrix
        correlation = actual_df / expected_df
        return correlation

    @property
    def speciation_data(self):
        """
        A dataframe containing the speciation of every solute at
        every trajectory frame. Indexed by frame and solute numbers.
        Columns are the solvent molecules and values are the number
        of solvent in the shell.
        """
        return self._speciation_df

    @property
    def speciation_fraction(self):
        """
        The fraction of shells of each type. Columns are the solvent
        molecules and values are the number of solvent in the shell.
        The final column is the fraction of total shell of that
        particular composition.
        """
        return self._speciation_fraction

    @property
    def solvent_co_occurrence(self):
        """
        The actual co-occurrence of solvents divided by the expected co-occurrence.
        In other words, given one molecule of solvent i in the shell, what is the
        probability of finding a solvent j relative to choosing a solvent at random
        from the pool of all coordinated solvents. This matrix will
        likely not be symmetric.
        """
        return self._solvent_co_occurrence
