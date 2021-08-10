import matplotlib.pyplot as plt
import pandas as pd

from MDAnalysis.analysis.base import AnalysisBase
from MDAnalysis.analysis.rdf import InterRDF
from MDAnalysis.lib.distances import capped_distance
import numpy as np
from solvation_analysis.rdf_parser import identify_solvation_cutoff
from solvation_analysis.analysis_library import (
    _Coordination,
    _Pairing,
    _Speciation,
    _SolvationData
)
from solvation_analysis.solvation import get_radial_shell, get_closest_n_mol, get_atom_group


class Solution(AnalysisBase):
    """
    The core class of the solvation module.
    """

    def __init__(
        self,
        solute,
        solvents,
        radii=None,
        rdf_kernel=None,
        kernel_kwargs=None,
        rdf_init_kwargs=None,
        rdf_run_kwargs=None,
        **kwargs,
    ):
        """
        Parameters
        ----------
            solute : AtomGroup
                the solute in the solutions
            solvents: dict
                a dictionary of names and atom groups. e.g. {"name_1": solvent_group_1,
                "name_2": solvent_group_2, ...}
            radii : dict
                an optional dictionary of solvation radii, any radii not
                given will be calculated. e.g. {"name_2": radius_2, "name_5": radius_5}
            rdf_kernel : function
                this function must take rdf bins and data as input and return
                a solvation radius as output. e.g. rdf_kernel(bins, data) -> 3.2. By default,
                the rdf_kernel is solvation_analysis.rdf_parser.identify_solvation_cutoff.
            kernel_kwargs : dict
                kwargs passed to rdf_kernel
            rdf_init_kwargs : dict
                kwargs passed to inner rdf initialization
            rdf_run_kwargs : dict
                kwargs passed to inner rdf run e.g. inner_rdf.run(**rdf_run_kwargs)
            kwargs : dict
                kwargs passed to AnalysisBase
        """
        super(Solution, self).__init__(solute.universe.trajectory, **kwargs)
        self.radii = {} if radii is None else radii
        self.kernel = identify_solvation_cutoff if rdf_kernel is None else rdf_kernel
        self.kernel_kwargs = {} if kernel_kwargs is None else kernel_kwargs
        self.rdf_init_kwargs = {"range": (0, 8.0)} if rdf_init_kwargs is None else rdf_init_kwargs
        self.rdf_run_kwargs = {} if rdf_run_kwargs is None else rdf_run_kwargs
        # TODO: save solute numbers somewhere
        self.solute = get_atom_group(solute)
        self.n_solute = len(self.solute)
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
        """
        Will plot the solvation radius on the rdf. If

        Parameters
        ----------
            bins : np.array
                the rdf bins
            data : np.array
                the rdf data
            radius : float
                the cutoff radius to draw on the plot

        Returns
        -------
            Matplotlib Figure, Matplotlib Axis
        """
        fig, ax = plt.subplots()
        ax.plot(bins, data, "b-", label="rdf")
        ax.axvline(radius, color="r", label="solvation radius")
        ax.set_xlabel("Radial Distance (A)")
        ax.set_ylabel("Probability Density")
        ax.legend()
        return fig, ax

    def _prepare(self):
        """
        This function identifies the solvation radii and saves the associated rdfs.
        """
        for name, solvent in self.solvents.items():
            # generate and save RDFs
            rdf = InterRDF(self.solute, solvent, **self.rdf_init_kwargs)
            # TODO: specify start and stop args
            rdf.run(**self.rdf_run_kwargs)
            bins, data = rdf.results.bins, rdf.results.rdf
            self.rdf_data[name] = (data, bins)
            # generate and save plots
            if name not in self.radii.keys():
                self.radii[name] = self.kernel(bins, data, **self.kernel_kwargs)
            fig, ax = self._plot_solvation_radius(bins, data, self.radii[name])
            ax.set_title(f"Solvation distance of {name}")
            self.rdf_plots[name] = fig, ax
        assert self.solvents.keys() == self.radii.keys(), "Radii missing."

    def _single_frame(self):
        """
        This function finds the solvation shells of each solute at a given time step.
        """
        # initialize empty arrays
        all_pairs_list = []
        all_dist_list = []
        all_tags_list = []
        # loop to find solvated atoms of each type
        for name, solvent in self.solvents.items():
            pairs, dist = capped_distance(
                self.solute.positions,
                solvent.positions,
                self.radii[name],
                box=self.u.dimensions,
            )
            # replace local ids with absolute ids
            pairs[:, 1] = solvent.ids[[pairs[:, 1]]]  # TODO: ids vs ix?
            # extend
            all_pairs_list.append(pairs)
            all_dist_list.append(dist)
            all_tags_list.append(np.full(len(dist), name))  # creating a name array
        all_pairs = np.concatenate(all_pairs_list, dtype=int)
        all_dist = np.concatenate(all_dist_list)
        all_tags = np.concatenate(all_tags_list)
        frame_length = len(all_pairs)
        all_frames = np.full(frame_length, self._ts.frame)
        # put the data into a data frame
        all_resid = self.u.atoms[all_pairs[:, 1]].resids
        solvation_data_np = np.column_stack(
            (all_frames, all_pairs[:, 0], all_pairs[:, 1], all_dist, all_tags, all_resid)
        )
        self.solvation_frames.append(solvation_data_np)

    def _conclude(self):
        """
        Instantiates the SolvationData class and several analysis classes.
        """
        solvation_data_np = np.vstack(self.solvation_frames)
        solvation_data_df = pd.DataFrame(
            solvation_data_np,
            columns=["frame", "solvated_atom", "atom_id", "dist", "res_name", "res_id"]
        )
        # clean up solvation_data df
        for column in ["frame", "solvated_atom", "atom_id", "dist", "res_id"]:
            solvation_data_df[column] = pd.to_numeric(solvation_data_df[column])
        solvation_data_dup = solvation_data_df.sort_values(["frame", "solvated_atom", "dist"])
        solvation_data = solvation_data_dup.drop_duplicates(["frame", "solvated_atom", "res_id"])
        self.solvation_data_dup = solvation_data_dup.set_index(["frame", "solvated_atom", "atom_id"])
        self.solvation_data = solvation_data.set_index(["frame", "solvated_atom", "atom_id"])
        # create analysis classes
        self.ion_speciation = _Speciation(self.solvation_data, self.n_frames, self.n_solute)
        self.ion_pairing = _Pairing(self.solvation_data, self.n_frames, self.n_solute)
        self.coordination_numbers = _Coordination(self.solvation_data, self.n_frames, self.n_solute)

    def map_step_to_index(self, traj_step):
        """
        This will map the given trajectory step to the nearest tested frame in the Solution.
        The index will select the analyzed trajectory step that is closest to but less
        than the given trajectory step.

        Parameters
        ----------
            traj_step : int
                the trajectory step of interest

            Returns
            -------
                index ; int
        """
        assert self.start <= traj_step <= self.stop, f"The traj_step {traj_step} " \
                                                     f"is not in the region covered by Solution."
        index = len(self.frames) - 1
        while traj_step < self.frames[index]:
            index -= 1
        return self.frames[index]

    def radial_shell(self, solute_index, radius, step=None):
        """
        Returns all molecules with atoms within the radius of the central species.
        (specifically, within the radius of the COM of central species).
        Thin wrapper around solvation.get_radial_shell

        Parameters
        ----------
            solute_index : int
                the index of the solute of interest
            radius : float or int
                radius used for atom selection
            step : int
                the step in the trajectory to perform selection at. Defaults to the
                current trajectory step.

        Returns
        -------
            AtomGroup
        """
        if step is not None:
            self.u.trajectory[step]
        return get_radial_shell(self.solute[solute_index], radius)

    def closest_n_mol(self, solute_index, n_mol, step=None, **kwargs):
        """
        Returns the closest n molecules to the central species. Optionally returns
        an array of their resids and an array of the distance of the closest atom
        in each molecule. Thin wrapper around solvation.get_closest_n_mol, see
        documentation for more detail.

        Parameters
        ----------
            solute_index : Atom, AtomGroup, Residue, or ResidueGroup
            n_mol : int
                The number of molecules to return
            step : int
                the step in the trajectory to perform selection at. Defaults to the
                current trajectory step.
            kwargs : passed to solvation.get_closest_n_mol

        Returns
        -------
            AtomGroup (molecules), np.Array (resids), np.Array (distances)

        """
        if step is not None:
            self.u.trajectory[step]
        return get_closest_n_mol(self.solute[solute_index], n_mol, **kwargs)

    def solvation_shell(self, solute_index, step):
        """
        Returns the solvation shell of the solute as an AtomGroup.

        Parameters
        ----------
            solute_index : Atom, AtomGroup, Residue, or ResidueGroup
            step : int
                the step in the trajectory to perform selection at. Defaults to the
                current trajectory step.

        Returns
        -------
            AtomGroup (molecules), np.Array (resids), np.Array (distances)

        """
        assert self.solvation_frames, "Solute.run() must be called first."
        # map to absolute frame index
        step = self.map_step_to_index(step)
        # select shell of interest
        shell = self.solvation_data.xs((step, solute_index), level=("frame", "solvated_atom"))
        # select shell AtomGroup
        ids = " ".join(shell["res_id"].astype(str))
        shell_group = self.u.select_atoms(f"resid {ids}")
        return shell_group
