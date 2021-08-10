import pandas as pd
import numpy as np


class Speciation:
    """
    A class for calculating and storing speciation information for solvents.

    Parameters
    ----------
        solvation_data : pandas.DataFrame
            The solvation data frame output by Solution
        n_frames : int
            The number of frames in solvation_data
        n_solutes : int
            The number of solutes in solvation_data

    Attributes
    ----------
        solvation_data : pandas.DataFrame
            The solvation data frame output by Solution
        n_frames : int
            The number of frames in solvation_data
        n_solutes : int
            The number of solutes in solvation_data

    Methods
    -------
        cluster_percent

        find_clusters
    """

    def __init__(self, solvation_data, n_frames, n_solutes):

        self.solvation_data = solvation_data
        self.n_frames = n_frames
        self.n_solutes = n_solutes
        self.speciation, self.speciation_percent = self._compute_speciation()

    def _compute_speciation(self):
        counts = self.solvation_data.groupby(["frame", "solvated_atom", "res_name"]).count()["res_id"]
        counts_re = counts.reset_index(["res_name"])
        speciation = counts_re.pivot(columns=["res_name"]).fillna(0).astype(int)
        res_names = speciation.columns.levels[1]
        speciation.columns = res_names
        sum_series = speciation.groupby(speciation.columns.to_list()).size()
        sum_sorted = sum_series.sort_values(ascending=False)
        speciation_percent = sum_sorted.reset_index().rename(columns={0: 'count'})
        speciation_percent['count'] = speciation_percent['count'] / (self.n_frames * self.n_solutes)
        return speciation, speciation_percent

    @classmethod
    def _average_speciation(cls, speciation_frames, solute_number, frame_number):
        averages = speciation_frames.sum(axis=1) / (solute_number * frame_number)
        return averages

    def cluster_percent(self, shell_dict):
        """
        This function should return the percent of clusters that exist with
        a particular composition.
        """
        query_list = [f"{name} == {str(count)}" for name, count in shell_dict.items()]
        query = " and ".join(query_list)
        query_counts = self.speciation_percent.query(query)
        return query_counts['count'].sum()

    def find_clusters(self, shell_dict):
        """
        This should return the step and solute # of all clusters of a particular composition.
        """
        query_list = [f"{name} == {str(count)}" for name, count in shell_dict.items()]
        query = " and ".join(query_list)
        query_counts = self.speciation.query(query)
        return query_counts


class Coordination:
    """
    A class for calculating and storing the coordination numbers of solvents.
    """

    def __init__(self, solvation_data, n_frames, n_solutes):
        """
        Parameters
        ----------
        solvation_data: the solvation data frame output by Solute
        """
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
    A class for analyzing pairing between the solute and another species.
    """

    def __init__(self, solvation_data, n_frames, n_solutes):
        """

        Parameters
        ----------
        solvation_data: the solvation data frame output by Solute
        """
        self.solvation_data = solvation_data
        self.n_frames = n_frames
        self.n_solutes = n_solutes
        self.percentage_dict = self._percentage_coordinated()

    def _percentage_coordinated(self):
        counts = self.solvation_data.groupby(["frame", "solvated_atom", "res_name"]).count()["res_id"]
        solutes_coordinated = counts.astype(bool).groupby(["res_name"]).sum() / (
            self.n_frames * self.n_solutes
        )  # average coordinated overall
        return solutes_coordinated.to_dict()
