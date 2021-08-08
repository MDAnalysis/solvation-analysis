import pandas as pd
import numpy as np


class _SolvationData:
    """
    A class for holding solvation data, this will slightly reprocess data and
    make it available in different forms.
    """

    def __init__(self, raw_solvation_frames):
        """
        Parameters
        ----------
        raw_solvation_frames: a list of tidy dataframes from the Solution class,
            encapsulating all solvent input
        """
        self.raw_solvation_frames = raw_solvation_frames
        self.solute_number = len(np.unique(self.raw_solvation_frames[0]["solvated_atom"]))
        self.frame_number = len(self.raw_solvation_frames)
        self.solvation_frames = [
            self._reindex_frame(frame) for frame in raw_solvation_frames
        ]
        self.solvation_frames_w_dup = [
            self._reindex_frame(frame, duplicates=True)
            for frame in raw_solvation_frames
        ]
        self.counts = self._accumulate_counts(self.solvation_frames)

    @classmethod
    def _reindex_frame(cls, raw_solvation_frame, duplicates=False):
        new_frame = raw_solvation_frame.sort_values(["solvated_atom", "dist"])
        if not duplicates:
            new_frame = new_frame.drop_duplicates(["solvated_atom", "res_id"])
        new_frame = new_frame.set_index(["solvated_atom", "atom_id"])
        return new_frame

    @classmethod
    def _accumulate_counts(cls, solvation_frames):
        counts_list = [
            frame.groupby(["solvated_atom", "res_name"]).count()["res_id"]
            for frame in solvation_frames
        ]
        counts_frame = pd.concat(
            counts_list, axis=1, names=["solvated_atom", "res_name"]
        )
        counts_frame = counts_frame.fillna(0)
        counts_frame.columns = range(len(solvation_frames))
        return counts_frame


class _IonSpeciation:
    """
    A class for calculating and storing speciation information for solvents.
    """

    def __init__(self, solvation_data):
        """

        Parameters
        ----------
        solvation_data: the solvation data frame output by Solute
        """
        self.solvation_frames = solvation_data.solvation_frames
        self.speciation_frames = self._accumulate_speciation(self.solvation_frames)
        self.average_speciation = self._average_speciation(
            self.speciation_frames, solvation_data.solute_number, solvation_data.frame_number
        )
        self.res_names = self._single_speciation_frame(
            self.solvation_frames[0], return_res_names=True
        )

    @classmethod
    def _single_speciation_frame(cls, frame, return_res_names=False):
        counts = frame.groupby(["solvated_atom", "res_name"]).count()["res_id"]
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


class _CoordinationNumber:
    """
    A class for calculating and storing the coordination numbers of solvents.
    """

    def __init__(self, solvation_data):
        """
        Parameters
        ----------
        solvation_data: the solvation data frame output by Solute
        """
        self.average_dict = self._average_cn(
            solvation_data.counts, solvation_data.solute_number, solvation_data.frame_number
        )

    @classmethod
    def _average_cn(cls, counts, solute_number, frame_number):
        mean_counts = counts.groupby(["res_name"]).sum().sum(axis=1) / (
            solute_number * frame_number
        )
        mean_dict = mean_counts.to_dict()
        return mean_dict


class _Pairing:
    """
    A class for analyzing pairing between the solute and another species.
    """

    def __init__(self, solvation_data):
        """

        Parameters
        ----------
        solvation_data: the solvation data frame output by Solute
        """
        self.percentage_dict = self._percentage_coordinated(
            solvation_data.counts, solvation_data.solute_number, solvation_data.frame_number
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
