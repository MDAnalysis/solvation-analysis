import matplotlib.pyplot as plt
import pandas as pd

from MDAnalysis.analysis.base import AnalysisBase
from MDAnalysis.analysis.rdf import InterRDF
from MDAnalysis.lib.distances import capped_distance
import numpy as np
from solvation_analysis.rdf_parser import identify_solvation_cutoff
from solvation_analysis.analysis_library import (
    Coordination,
    Pairing,
    Speciation,
)
from solvation_analysis.selection import get_radial_shell, get_closest_n_mol, get_atom_group


class Solution(AnalysisBase):
    """
    The core class of the solvation module.

    Parameters
    ----------
        solute : AtomGroup
            the solute in the solutions
        solvents: dict
            a dictionary of names and atom groups. e.g. {"name_1": solvent_group_1,
            "name_2": solvent_group_2, ...}
        radii : dict, optional
            an optional dictionary of solvation radii, any radii not
            given will be calculated. e.g. {"name_2": radius_2, "name_5": radius_5}
        rdf_kernel : function, optional
            this function must take rdf bins and data as input and return
            a solvation radius as output. e.g. rdf_kernel(bins, data) -> 3.2. By default,
            the rdf_kernel is solvation_analysis.rdf_parser.identify_solvation_cutoff.
        kernel_kwargs : dict, optional
            kwargs passed to rdf_kernel
        rdf_init_kwargs : dict, optional
            kwargs passed to inner rdf initialization
        rdf_run_kwargs : dict, optional
            kwargs passed to inner rdf run e.g. inner_rdf.run(**rdf_run_kwargs)
        kwargs : dict, optional
            kwargs passed to AnalysisBase

    Attributes
    ----------
    u : Universe
        An MDAnalysis Universe object
    n_solute : int
        number of solute atoms
    radii : dict
        a dictionary of solvation radii for each solvent
        e.g. {"name_2": radius_2, "name_2": radius_2, ...}
    rdf_data : dict
        a dictionary of rdf data, keys are solvent names and values
        are (bins, data) tuples.
    solvation_data : pandas.DataFrame
        a dataframe of solvation data with columns "frame", "solvated_atom", "atom_id",
        "dist", "res_name", and "res_id". If multiple entries share a frame, solvated_atom,
        and atom_id, all but the the closest atom is dropped.
    solvation_data_dup : pandas.DataFrame
        a dataframe of solvation data with columns "frame", "solvated_atom", "atom_id",
        "dist", "res_name", and "res_id". If multiple entries share a frame, solvated_atom,
        and atom_id, all atoms are kept.
    pairing : Pairing object
        An analysis_library.Pairing object instantiated from solvation_data.
    coordination : Coordination object
        An analysis_library.Coordination object instantiated from solvation_data.
    speciation : Speciation object
        An analysis_library.Speciation object instantiated from solvation_data.
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
        self.rdf_data = {}
        self.solvation_data = None
        self.solvation_data_dup = None
        self.speciation = None
        self.pairing = None
        self.coordination = None
        self.solvation_frames = []

    def _prepare(self):
        """
        This function identifies the solvation radii and saves the associated rdfs.
        """
        for name, solvent in self.solvents.items():
            # generate and save RDFs
            rdf = InterRDF(self.solute, solvent, **self.rdf_init_kwargs)
            rdf.run(**self.rdf_run_kwargs)
            bins, data = rdf.results.bins, rdf.results.rdf
            self.rdf_data[name] = (bins, data)
            # generate and save plots
            if name not in self.radii.keys():
                self.radii[name] = self.kernel(bins, data, **self.kernel_kwargs)
        assert self.solvents.keys() == self.radii.keys(), "Radii missing."

    def _single_frame(self):
        """
        This function finds the solvation shells of each solute at a given time step.
        """
        # initialize empty lists
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
        self.speciation = Speciation(self.solvation_data, self.n_frames, self.n_solute)
        self.pairing = Pairing(self.solvation_data, self.n_frames, self.n_solute)
        self.coordination = Coordination(self.solvation_data, self.n_frames, self.n_solute)

    @staticmethod
    def _plot_solvation_radius(bins, data, radius):
        """
        Will plot the solvation radius on the rdf from bins, data, and a radius.
        Includes a vertical line at the radius of interest.

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
            Matplotlib Figure, Matplotlib Axes
        """
        fig, ax = plt.subplots()
        ax.plot(bins, data, "b-", label="rdf")
        ax.axvline(radius, color="r", label="solvation radius")
        ax.set_xlabel("Radial Distance (A)")
        ax.set_ylabel("Probability Density")
        ax.legend()
        return fig, ax

    def plot_solvation_radius(self, res_name):
        """
        Will plot the rdf of a solvent molecule, specified by resname.
        Includes a vertical line at the radius of interest.

        Parameters
        ----------
        res_name : str
            the name of the residue of interest, as written in the solvents dict

        Returns
        -------
            Matplotlib Figure, Matplotlib Axes
        """
        bins, data = self.rdf_data[res_name]
        fig, ax = self._plot_solvation_radius(bins, data, self.radii[res_name])
        ax.set_title(f"Solvation distance of {res_name}")
        return fig, ax

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
            step : int, optional
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
            step : int, optional
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

    def solvation_shell(self, solute_index, step, as_df=False, remove_mols=None, closest_n_only=None):
        """
        Returns the solvation shell of the solute as an AtomGroup.

        Parameters
        ----------
            solute_index : Atom, AtomGroup, Residue, or ResidueGroup
            step : int
                the step in the trajectory to perform selection at. Defaults to the
                current trajectory step.
            as_df : boolean, default=False
                if true, this function will return a DataFrame representing the shell
                instead of a AtomGroup.
            remove_mols : dict, optional
                remove_dict lets you remove specific residues from the final shell.
                It should be a dict of molnames and ints e.g. {'mol1': n, 'mol2', m}.
                It will remove up to n of mol1 and up to m of mol2. So if the dict is
                {'mol1': 1, 'mol2', 1} and the shell has 4 mol1 and 0 mol2,
                solvation_shell will return a shell with 3 mol1 and 0 mol2.
            closest_n_only : int, optional
                if given, only the closest n residues will be included

        Returns
        -------
            AtomGroup or DataFrame

        """
        assert self.solvation_frames, "Solute.run() must be called first."
        assert step in self.frames, "The requested step must be one of an " \
                                    "analyzed steps in self.frames."
        remove_mols = {} if remove_mols is None else remove_mols
        # select shell of interest
        shell = self.solvation_data.xs((step, solute_index), level=("frame", "solvated_atom"))
        # remove mols
        for mol_name, n_remove in remove_mols.items():
            # first, filter for only mols of type mol_name
            is_mol = shell.res_name == mol_name
            res_ids = shell[is_mol].res_id
            mol_count = len(res_ids)
            n_remove = min(mol_count, n_remove)
            # then truncate resnames to remove mols
            remove_ids = res_ids[(mol_count - n_remove):]
            # then apply to original shell
            remove = shell.res_id.isin(remove_ids)
            shell = shell[np.invert(remove)]
        # filter based on length
        if closest_n_only:
            assert closest_n_only > 0, "closest_n_only must be at least 1"
            closest_n_only = min(len(shell), closest_n_only)
            shell = shell[0: closest_n_only]
        if as_df:
            return shell
        else:
            return self.resids_to_atom_group(shell["res_id"], solute_index=solute_index)

    def resids_to_atom_group(self, ids, solute_index=None):
        """

        Parameters
        ----------
        ids : np.array[int]
            an array of res ids
        solute_index : int, optional
            if given, will include the solute with solute_index

        Returns

        -------

        """
        ids = " ".join(ids.astype(str))
        atoms = self.u.select_atoms(f"resid {ids}")
        if solute_index is not None:
            atoms = atoms | self.solute[solute_index]
        return atoms
