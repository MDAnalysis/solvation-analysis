import pandas as pd


class _IonSpeciation:
    def __init__(self, solvation_data):
        self.solvation_frames = solvation_data


class _CoordinationNumbers:
    def __init__(self, solvation_data):
        self.solvation_data = solvation_data
        self.counts = self._accumulate_counts(self.solvation_data)
        self.average_dict = self._average(self.counts)

    @classmethod
    def _single_frame(cls, frame):
        frame = frame.drop_duplicates(["solvated_atom", "res_id"])
        counts = frame.groupby(["solvated_atom", "res_name"]).count()["res_id"]
        averages = counts.groupby(["res_name"]).mean()
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
    def __init__(self, solvation_data):
        self.solvation_frames = solvation_data
