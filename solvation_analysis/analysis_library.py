import pandas as pd


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
        self.solvation_frames = solvation_data


class _CoordinationNumbers:
    """
    A class for calculating and storing the coordination numbers of solvents.
    """
    def __init__(self, solvation_data):
        """
        Parameters
        ----------
        solvation_data: the solvation data frame output by Solute
        """
        self.solvation_data = solvation_data
        self.counts = self._accumulate_counts(self.solvation_data)
        self.average_dict = self._average(self.counts)

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
        counts_frame.columns = range(len(frames))
        return counts_frame

    @classmethod
    def _average(cls, counts):
        mean_counts = counts.groupby(["res_name"]).mean().mean(axis=1)
        mean_dict = mean_counts.to_dict()
        return mean_dict


class _IonPairing:
    """
    A class for analyzing pairing between the solute and another species.
    """
    def __init__(self, solvation_data):
        """

        Parameters
        ----------
        solvation_data: the solvation data frame output by Solute
        """
        self.solvation_frames = solvation_data
