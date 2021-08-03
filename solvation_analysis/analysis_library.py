import pandas as pd
import numpy as np


class _SolutionAnalysis:
    """
    An interface for analyzing the solvation data from Solute.
    """

    def __init__(self, solvation_data):
        """
        Parameters
        ----------
        solvation_data: the solvation data frame output by Solute
        """
        self.solvation_data = solvation_data
        self.solute_number = len(np.unique(solvation_data[0]["solvated_atom"]))
        self.frame_number = len(solvation_data)
        self.counts = self._accumulate_counts(self.solvation_data)

    @classmethod
    def _single_frame(cls, frame):
        frame = frame.drop_duplicates(["solvated_atom", "res_id"])
        counts = frame.groupby(["solvated_atom", "res_name"]).count()["res_id"]
        return counts

    @classmethod
    def _accumulate_counts(cls, frames):
        counts_list = [cls._single_frame(frame) for frame in frames]
        counts_frame = pd.concat(
            counts_list, axis=1, names=["solvated_atom", "res_name"]
        )
        counts_frame = counts_frame.fillna(0)
        counts_frame.columns = range(len(frames))
        return counts_frame


class _IonSpeciation(_SolutionAnalysis):
    """
    A class for calculating and storing speciation information for solvents.
    """

    def __init__(self, solvation_data):
        """

        Parameters
        ----------
        solvation_data: the solvation data frame output by Solute
        """
        super().__init__(solvation_data)
        self.solvation_frames = solvation_data
        self.speciation_frames = self._accumulate_speciation(self.solvation_frames)
        self.average_speciation = self._average_speciation(
            self.speciation_frames, self.solute_number, self.frame_number
        )
        self.res_names = self._single_speciation_frame(
            self.solvation_frames[0], return_res_names=True
        )

    @classmethod
    def _single_speciation_frame(cls, frame, return_res_names=False):
        counts = cls._single_frame(frame)
        counts_re = counts.reset_index(["res_name"])
        pivoted = counts_re.pivot(columns=["res_name"]).fillna(0).astype(int)
        res_names = pivoted.columns.levels[1]
        pivoted.columns = res_names
        speciation_counts = pivoted.groupby(pivoted.columns.to_list()).size()
        if return_res_names:
            return res_names
        return speciation_counts

    @classmethod
    def _accumulate_speciation(cls, frames):
        speciation_list = [cls._single_speciation_frame(frame) for frame in frames]
        speciation = pd.concat(speciation_list, axis=1).fillna(0)
        return speciation

    @classmethod
    def _average_speciation(cls, speciation_frames, solute_number, frame_number):
        averages = speciation_frames.sum(axis=1) / (solute_number * frame_number)
        return averages

    def check_cluster_percent(self, shell_tuple):
        """
        This function should return the percent of clusters that exist with
        a particular composition.
        """
        if shell_tuple in self.average_speciation.index.to_list():
            return self.average_speciation[shell_tuple]
        else:
            return 0

    def find_clusters(self, shell_tuple):
        """
        This should return the step and solute # of all clusters of a particular composition.
        """
        return


class _CoordinationNumber(_SolutionAnalysis):
    """
    A class for calculating and storing the coordination numbers of solvents.
    """

    def __init__(self, solvation_data):
        """
        Parameters
        ----------
        solvation_data: the solvation data frame output by Solute
        """
        super().__init__(solvation_data)
        self.average_dict = self._average_cn(
            self.counts, self.solute_number, self.frame_number
        )

    @classmethod
    def _average_cn(cls, counts, solute_number, frame_number):
        mean_counts = counts.groupby(["res_name"]).sum().sum(axis=1) / (
            solute_number * frame_number
        )
        mean_dict = mean_counts.to_dict()
        return mean_dict


class _Pairing(_SolutionAnalysis):
    """
    A class for analyzing pairing between the solute and another species.
    """
    def __init__(self, solvation_data):
        """

        Parameters
        ----------
        solvation_data: the solvation data frame output by Solute
        """
        super().__init__(solvation_data)
        self.percentage_dict = self._percentage_coordinated(
            self.counts, self.solute_number, self.frame_number
        )

    @classmethod
    def _percentage_coordinated(cls, counts, solute_number, frame_number):
        solutes_coordinated = counts.astype(bool).sum(  # coordinated or not
            axis=1
        ).groupby(  # average number coordinated, per solute
            level=1
        ).sum() / (
            solute_number * frame_number
        )  # average coordinated overall
        return solutes_coordinated
