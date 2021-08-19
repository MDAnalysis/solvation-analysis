"""
================
Analysis Library
================
:Author: Orion Cohen
:Year: 2021
:Copyright: GNU Public License v3

Analysis library defines a variety of classes that analyze different aspects of solvation.
These classes are all instantiated with the solvation_data (pandas.DataFrame) generated
from the Solution class.

While the classes in analysis_library can be used in isolation, they are meant to be used
as attributes of the Solution class. This makes instantiating them and calculating the
solvation data a non-issue.
"""

import pandas as pd
import numpy as np


class Speciation:
    """
    Calculate the solvation shells of every solute.

    Speciation organizes the solvation data by the type of residue
    coordinated with the central solvent. It collects this information in a
    pandas.DataFrame indexed by the frame and solute number. Each column is
    one of the solvents in the res_name column of the solvation data. The
    column value is how many residue of that type are in the solvation shell.

    Speciation provides the speciation of each solute in the speciation
    attribute, it also calculates the percentage of each unique
    shell and makes it available in the speciation_percent attribute.

    Additionally, there are methods for finding solvation shells of
    interest and computing how common certain shell configurations are.

    Parameters
    ----------
    solvation_data : pandas.DataFrame
        The solvation data frame output by Solution.
    n_frames : int
        The number of frames in solvation_data.
    n_solutes : int
        The number of solutes in solvation_data.

    Attributes
    ----------
    speciation_data : pandas.DataFrame
        a dataframe containing the speciation of every li ion at
        every trajectory frame. Indexed by frame and solute numbers.
        Columns are the solvent molecules and values are the number
        of solvent in the shell.
    speciation_percent : pandas.DataFrame
        the percentage of shells of each type. Columns are the solvent
        molecules and and values are the number of solvent in the shell.
        The final column is the percentage of total shell of that
        particular composition.
    """

    def __init__(self, solvation_data, n_frames, n_solutes):
        self.solvation_data = solvation_data
        self.n_frames = n_frames
        self.n_solutes = n_solutes
        self.speciation_data, self.speciation_percent = self._compute_speciation()

    def _compute_speciation(self):
        counts = self.solvation_data.groupby(["frame", "solvated_atom", "res_name"]).count()["res_id"]
        counts_re = counts.reset_index(["res_name"])
        speciation_data = counts_re.pivot(columns=["res_name"]).fillna(0).astype(int)
        res_names = speciation_data.columns.levels[1]
        speciation_data.columns = res_names
        sum_series = speciation_data.groupby(speciation_data.columns.to_list()).size()
        sum_sorted = sum_series.sort_values(ascending=False)
        speciation_percent = sum_sorted.reset_index().rename(columns={0: 'count'})
        speciation_percent['count'] = speciation_percent['count'] / (self.n_frames * self.n_solutes)
        return speciation_data, speciation_percent

    @classmethod
    def _average_speciation(cls, speciation_frames, solute_number, frame_number):
        averages = speciation_frames.sum(axis=1) / (solute_number * frame_number)
        return averages

    def shell_percent(self, shell_dict):
        """
        Calculate the percentage of shells matching shell_dict.

        This function computes the percent of solvation shells that exist with a particular
        composition. The composition is specified by the shell_dict. The percent
        will be of all shells that match that specification.

        Attributes
        ----------
        shell_dict : dict of {str: int}
            a specification for a shell composition. Keys are residue names (str)
            and values are the number of desired residues. e.g. if shell_dict =
            {'mol1': 4} then the function will return the percentage of shells
            that have 4 mol1. Note that this may include shells with 4 mol1 and
            any number of other solvents. To specify a shell with 4 mol1 and nothing
            else, enter a dict such as {'mol1': 4, 'mol2': 0, 'mol3': 0}.

        Returns
        -------
        float
            the percentage of shells

        Examples
        --------

         .. code-block:: python

            # first define Li, BN, and FEC AtomGroups
            >>> solution = Solution(Li, {'BN': BN, 'FEC': FEC, 'PF6': PF6})
            >>> solution.run()
            >>> solution.speciation.shell_percent({'BN': 4, 'PF6': 1})
            0.0898
        """
        query_list = [f"{name} == {str(count)}" for name, count in shell_dict.items()]
        query = " and ".join(query_list)
        query_counts = self.speciation_percent.query(query)
        return query_counts['count'].sum()

    def find_shells(self, shell_dict):
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

    The coordination numbers are made available as an average over the whole
    simulation and by frame.

    Parameters
    ----------
    solvation_data : pandas.DataFrame
        The solvation data frame output by Solution.
    n_frames : int
        The number of frames in solvation_data.
    n_solutes : int
        The number of solutes in solvation_data.

    Attributes
    ----------
    cn_dict : dict of {str: float}
        a dictionary where keys are residue names (str) and values are the
        average coordination number of that residue (float).
    cn_by_frame : pd.DataFrame
        a dictionary tracking the average coordination number of each
        residue across frames.

    Examples
    --------

     .. code-block:: python

        # first define Li, BN, and FEC AtomGroups
        >>> solution = Solution(Li, {'BN': BN, 'FEC': FEC, 'PF6': PF6})
        >>> solution.run()
        >>> solution.coordination.cn_dict
        {'BN': 4.328, 'FEC': 0.253, 'PF6': 0.128}

    """

    def __init__(self, solvation_data, n_frames, n_solutes):
        self.solvation_data = solvation_data
        self.n_frames = n_frames
        self.n_solutes = n_solutes
        self.cn_dict, self.cn_by_frame = self._average_cn()

    def _average_cn(self):
        counts = self.solvation_data.groupby(["frame", "solvated_atom", "res_name"]).count()["res_id"]
        cn_series = counts.groupby(["res_name", "frame"]).sum() / (
            self.n_solutes * self.n_frames
        )
        cn_by_frame = cn_series.unstack()
        cn_dict = cn_series.groupby(["res_name"]).sum().to_dict()
        return cn_dict, cn_by_frame


class Pairing:
    """
    Calculate the percent of solutes that are coordinated with each solvent.

    The pairing percentage is the percent of solutes that are coordinated with
    ANY solvent with matching type. So if the pairing of mol1 is 0.5, then 50% of
    solutes are coordinated with at least 1 mol1.

    The pairing percentages are made available as an average over the whole
    simulation and by frame.

    Parameters
    ----------
    solvation_data : pandas.DataFrame
        The solvation data frame output by Solution.
    n_frames : int
        The number of frames in solvation_data.
    n_solutes : int
        The number of solutes in solvation_data.

    Attributes
    ----------
    pairing_dict : dict of {str: float}
        a dictionary where keys are residue names (str) and values are the
        percentage of solutes that contain that residue (float).
    pairing_by_frame : pd.DataFrame
        a dictionary tracking the average percentage of each
        residue across frames.

    Examples
    --------

     .. code-block:: python

        # first define Li, BN, and FEC AtomGroups
        >>> solution = Solution(Li, {'BN': BN, 'FEC': FEC, 'PF6': PF6})
        >>> solution.run()
        >>> solution.pairing.pairing_dict
        {'BN': 1.0, 'FEC': 0.210, 'PF6': 0.120}
    """

    def __init__(self, solvation_data, n_frames, n_solutes):
        self.solvation_data = solvation_data
        self.n_frames = n_frames
        self.n_solutes = n_solutes
        self.pairing_dict, self.pairing_by_frame = self._percentage_coordinated()

    def _percentage_coordinated(self):
        counts = self.solvation_data.groupby(["frame", "solvated_atom", "res_name"]).count()["res_id"]
        pairing_series = counts.astype(bool).groupby(["res_name", "frame"]).sum() / (
            self.n_solutes
        )  # average coordinated overall
        pairing_by_frame = pairing_series.unstack()
        pairing_normalized = pairing_series / self.n_frames
        pairing_dict = pairing_normalized.groupby(["res_name"]).sum().to_dict()
        return pairing_dict, pairing_by_frame
