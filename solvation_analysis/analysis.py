# from abc import ABC
from abc import ABC

import matplotlib.pyplot as plt
import pandas as pd

from MDAnalysis.analysis.base import AnalysisBase
from MDAnalysis.analysis.rdf import InterRDF
from MDAnalysis.analysis import distances
import numpy as np
import xarray as xr
from solvation_analysis.rdf_parser import identify_solvation_cutoff


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
        all_resid = np.empty(0)
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
        all_resid = self.u.atoms[all_pairs[:, 1]].resids
        data_chonk = np.column_stack((all_pairs[:, 0], all_pairs[:, 1], all_dist, all_tags, all_resid))
        data_slice = pd.DataFrame(data_chonk, columns=['li_atom', 'solv_atom_id', 'dist', 'res_name', 'res_id'])
        # select for only pairs including solvent
        # selection = np.in1d(all_pairs[:, 0], self.solute.ids)

        print("hi")

        # REQUIRED
        # Called after the trajectory is moved onto each new frame.
        # store result of `some_function` for a single frame
        # self.result.append(some_function(self._ag, self._parameter))

    # def _conclude(self):
    #     # OPTIONAL
    #     # Called once iteration on the trajectory is finished.
    #     # Apply normalisation and averaging to results here.
    #     self.result = np.asarray(self.result) / np.sum(self.result)
    #     self.ion_speciation = IonSpeciation(self.solute, self.solvents)
    #     self.ion_pairing = IonPairing(self.solute, self.solvents)
    #     self.coordination_numbers = CoordinationNumber(self.solute, self.solvents)


class IonSpeciation(AnalysisBase):
    def __init__(self, solute, solvent, **kwargs):
        super(IonSpeciation, self).__init__(solute.universe.trajectory, **kwargs)
        self._parameter = solvent
        self._ag = solute

    def _prepare(self):
        # OPTIONAL
        # Called before iteration on the trajectory has begun.
        # Data structures can be set up at this time
        self.result = []

    def _single_frame(self):
        # REQUIRED
        # Called after the trajectory is moved onto each new frame.
        # store result of `some_function` for a single frame
        self.result.append(some_function(self._ag, self._parameter))

    def _conclude(self):
        # OPTIONAL
        # Called once iteration on the trajectory is finished.
        # Apply normalisation and averaging to results here.
        self.result = np.asarray(self.result) / np.sum(self.result)


class IonPairing(AnalysisBase):
    def __init__(self, solute, solvent, **kwargs):
        super(IonPairing, self).__init__(solute.universe.trajectory, **kwargs)
        self._parameter = solvent
        self._ag = solute

    def _prepare(self):
        # OPTIONAL
        # Called before iteration on the trajectory has begun.
        # Data structures can be set up at this time
        self.result = []

    def _single_frame(self):
        # REQUIRED
        # Called after the trajectory is moved onto each new frame.
        # store result of `some_function` for a single frame
        self.result.append(some_function(self._ag, self._parameter))

    def _conclude(self):
        # OPTIONAL
        # Called once iteration on the trajectory is finished.
        # Apply normalisation and averaging to results here.
        self.result = np.asarray(self.result) / np.sum(self.result)


class CoordinationNumber(AnalysisBase):
    def __init__(self, solute, solvent, **kwargs):
        super(CoordinationNumber, self).__init__(solute.universe.trajectory, **kwargs)
        self._parameter = solvent
        self._ag = solute

    def _prepare(self):
        # OPTIONAL
        # Called before iteration on the trajectory has begun.
        # Data structures can be set up at this time
        self.result = []

    def _single_frame(self):
        # REQUIRED
        # Called after the trajectory is moved onto each new frame.
        # store result of `some_function` for a single frame
        self.result.append(some_function(self._ag, self._parameter))

    def _conclude(self):
        # OPTIONAL
        # Called once iteration on the trajectory is finished.
        # Apply normalisation and averaging to results here.
        self.result = np.asarray(self.result) / np.sum(self.result)
