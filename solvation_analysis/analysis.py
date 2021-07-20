# from abc import ABC
from abc import ABC

import matplotlib.pyplot as plt
import pandas as pd

from MDAnalysis.analysis.base import AnalysisBase
from MDAnalysis.analysis.rdf import InterRDF
from MDAnalysis.analysis import distances
import numpy as np
from solvation_analysis.rdf_parser import identify_solvation_cutoff
from solvation_analysis.analysis_library import (
    _CoordinationNumbers,
    _IonPairing,
    _IonSpeciation,
)


def some_function():
    return


class Solute(AnalysisBase):
    def __init__(
        self, solute, solvents, radii=None, kernel=None, kernel_kwargs=None, **kwargs
    ):
        super(Solute, self).__init__(solute.universe.trajectory, **kwargs)
        self.radii = {} if radii is None else radii
        self.kernel = identify_solvation_cutoff if kernel is None else kernel
        # if not kernel:
        #     self.kernel = identify_solvation_cutoff
        self.kernel_kwargs = {} if kernel_kwargs is None else radii
        # if not kernel_kwargs:
        #     self.kernel_kwargs = {}

        self.solute = solute
        self.solvents = solvents
        self.u = self.solute.universe
        self.rdf_plots = {}
        self.rdf_data = {}
        self.ion_speciation = None
        self.ion_pairing = None
        self.coordination_numbers = None
        self.solvation_frames = []

    @classmethod
    def _plot_solvation_radius(cls, bins, data, radius):
        fig, ax = plt.subplots()
        ax.plot(bins, data, "b-", label="rdf")
        ax.axvline(radius, color="r", label="solvation radius")
        ax.set_xlabel("Radial Distance (A)")
        ax.set_ylabel("Probability Density")
        ax.legend()
        return fig, ax

    def run_prepare(self):
        for name, solvent in self.solvents.items():
            # generate and save RDFs
            rdf = InterRDF(self.solute, solvent, range=(0.0, 8.0))
            # TODO: specify start and stop args
            rdf.run()
            bins, data = rdf.results.bins, rdf.results.rdf
            self.rdf_data[name] = (data, bins)
            # generate and save plots
            if name not in self.radii.keys():
                self.radii[name] = self.kernel(bins, data, **self.kernel_kwargs)
            fig, ax = self._plot_solvation_radius(bins, data, self.radii[name])
            ax.set_title(f"Solvation distance of {name}")
            self.rdf_plots[name] = fig, ax

    def _prepare(self):
        assert self.solvents.keys() == self.radii.keys(), "Radii missing."
        # columns: solute #, atom id, distance, solvent name, res id

    def _single_frame(self):
        # initialize empty arrays
        all_pairs = np.empty((0, 2), dtype=np.int)
        all_dist = np.empty(0)
        all_tags = np.empty(0)
        for name, solvent in self.solvents.items():
            pairs, dist = distances.capped_distance(
                self.solute.positions,
                solvent.positions,
                self.radii[name],
                box=self.u.dimensions,
            )
            # replace local ids with absolute ids
            pairs[:, 1] = solvent.ids[[pairs[:, 1]]]  # TODO: ids vs ix?
            # extend
            all_pairs = np.concatenate((all_pairs, pairs))
            all_dist = np.concatenate((all_dist, dist))
            all_tags = np.concatenate((all_tags, np.full(len(dist), name)))
        # put the data into a data frame
        all_resid = self.u.atoms[all_pairs[:, 1]].resids
        solvation_data_np = np.column_stack(
            (all_pairs[:, 0], all_pairs[:, 1], all_dist, all_tags, all_resid)
        )
        solvation_data_pd = pd.DataFrame(
            solvation_data_np,
            columns=["solvated_atom", "atom_id", "dist", "res_name", "res_id"],
        )
        self.solvation_frames.append(solvation_data_pd)

    def _conclude(self):
        # OPTIONAL

        self.ion_speciation = _IonSpeciation(self.solvation_frames)
        self.ion_pairing = _IonPairing(self.solvation_frames)
        self.coordination_numbers = _CoordinationNumbers(self.solvation_frames)
