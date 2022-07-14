"""
========
Solution
========
:Author: Orion Cohen
:Year: 2021
:Copyright: GNU Public License v3

The solvation_analysis module is centered around the Solution class, which defines
solvation as coordination of a central solute with surrounding solvents. The Solution
class provides a convenient interface for specifying a solute and solvents, calculating
their solvation radii, and collecting the solvation shells of each solute into a
pandas.DataFrame for convenient analysis.

Solution uses the solvation data to instantiate each of the analysis classes in
the analysis_library as attributes. Creating a convenient interface for more in
depth analysis of specific aspects of solvation.

Solution also provides several functions to select a particular solute and its solvation
shell, returning an AtomGroup for visualization or further analysis.
"""


import matplotlib.pyplot as plt
import pandas as pd
import warnings

from MDAnalysis.analysis.base import AnalysisBase
from MDAnalysis.analysis.rdf import InterRDF
from MDAnalysis.lib.distances import capped_distance
import numpy as np
from solvation_analysis.rdf_parser import identify_cutoff_poly, identify_cutoff_scipy
from solvation_analysis.analysis_library import (
    Coordination,
    Pairing,
    Speciation,
    Residence,
    Networking,
)
from solvation_analysis.selection import get_radial_shell, get_closest_n_mol, get_atom_group


class Solution(AnalysisBase):
    """
    Analyze the solvation structure of a solution.

    Solution finds the coordination between the solute and each solvent
    and collects that information in a pandas.DataFrame (solvation_data)
    for convenient analysis. The names provided in the solvents dictionary
    are used throughout the class.

    First, Solution calculates the RDF between the solute and each solvent and
    uses it to identify the radius of the first solvation shell. Radii can
    instead be supplied with the radii parameter. After Solution.run() is
    called, these radii can be queried with the plot_solvation_radius method.

    Second, Solution finds all atoms in the first solvation shell, using
    the cutoff radii for each solvent. For each coordinating atom the id,
    residue id, and distance from the solute are saved in solvation_data.
    This analysis is repeated for each solute at every frame in the
    analysis and the data is compiled into a pandas.DataFrame and indexed
    by frame, solute number, and atom id.

    Finally, Solution instantiates Speciation, Coordination, and Pairing
    objects from the solvation_data, providing a convenient interface to
    further analysis.

    Note: Atom and Residue ids (1-based) are returned, not ix (0-based).
    This aligns with the MDAnalysis selection language.

    Parameters
    ----------
    solute : MDAnalysis.AtomGroup
        the solute in the solutions
    solvents: dict of {str: MDAnalysis.AtomGroup}
        a dictionary of solvent names and associated MDAnalysis.AtomGroups.
        e.g. {"name_1": solvent_group_1,"name_2": solvent_group_2, ...}
    radii : dict of {str: float}, optional
        an optional dictionary of solvent names and associated solvation radii
        e.g. {"name_2": radius_2, "name_5": radius_5} Any radii not given will
        be calculated. The solvent names should match the solvents parameter.
    solvent_counts : dict of {str: int}, optional
        an optional dictionary of solvent counts e.g. the number of solvent. Any
        solvents not included will be set equal to the number of residues in the
        solvents AtomGroup.
    rdf_kernel : function, optional
        this function must take RDF bins and data as input and return
        a solvation radius as output. e.g. rdf_kernel(bins, data) -> 3.2. By default,
        the rdf_kernel is solvation_analysis.rdf_parser.identify_solvation_cutoff.
    kernel_kwargs : dict, optional
        kwargs passed to rdf_kernel
    rdf_init_kwargs : dict, optional
        kwargs passed to the initialization of the MDAnalysis.InterRDF used to plot
        the solute-solvent RDFs. By default, ``range`` will be set to (0, 7.5).
    rdf_run_kwargs : dict, optional
        kwargs passed to the internal MDAnalysis.InterRDF.run() command
        e.g. ``inner_rdf.run(**rdf_run_kwargs)``. By default, step, start, and
        stop will use any kwargs provided to ``solution.run(**kwargs)``.
    solute_name: str, optional
        the name of the solute, used for labeling.
    analysis_classes : List[str], optional
        a list of the analysis classes to be instantiated, current options are:
        "pairing", "coordination", "speciation", "residence", and "networking".
        By default, only "pairing", "coordination", and "residence" are instantiated.
        If networking is included, the networking_solvents kwarg must be specified.
    networking_solvents : str, optional
        see the `solvents` parameter of the Networking class.
    verbose : bool, optional
       Turn on more logging and debugging, default ``False``

    Attributes
    ----------
    u : Universe
        An MDAnalysis.Universe object
    n_solute : int
        number of solute atoms
    radii : dict
        a dictionary of solvation radii for each solvent
        e.g. {"name_2": radius_2, "name_2": radius_2, ...}
    rdf_data : dict
        a dictionary of RDF data, keys are solvent names and values
        are (bins, data) tuples.
    solvation_data : pandas.DataFrame
        a dataframe of solvation data with columns "frame", "solvated_atom", "atom_ix",
        "dist", "res_name", and "res_ix". If multiple entries share a frame, solvated_atom,
        and atom_ix, all but the closest atom is dropped.
    solvation_data_dup : pandas.DataFrame
        a dataframe of solvation data with columns "frame", "solvated_atom", "atom_ix",
        "dist", "res_name", and "res_ix". If multiple entries share a frame, solvated_atom,
        and atom_ix, all atoms are kept.
    solute_res_ix : np.array
        a numpy array of the residue indices of every solute.
    solute_atom_ix : np.array
        a numpy array of the atom indices of every solute.
    res_name_map : pd.Series
        a map from residue indices in the Universe to solvent and solute names from
        the solution. For example, if the first residue in the universe was in
        ``self.solvent['res_one']``, then ``res_name_map[0] == 'res_one'``.
    pairing : analysis_library.Pairing (optional)
        pairing analyzes the fraction solutes and solvents that are coordinated.
    coordination : analysis_library.Coordination (optional)
        coordination analyses the coordination numbers of solvents and which
        solvent atoms are coordinated.
    speciation : analysis_library.Speciation (optional)
        speciation provides an interface for finding and selecting the solvation shells
        surrounding each solute.
    residence : analysis_library.Residence (optional)
        residence calculates the residence times of each solvent on the solute. Only
        instantiated if 'residence' is included in the analysis_classes kwarg.
    networking : analysis_library.Networking (optional)
        networking analyses the connectivity of solute-solvent networks. Only instantiated
        if 'networking' is included in the analysis_classes kwarg. the networking_solvents
        kwarg must be specified.
    """

    def __init__(
        self,
        solute,
        solvents,
        radii=None,
        solvent_counts=None,
        rdf_kernel=None,
        kernel_kwargs=None,
        rdf_init_kwargs=None,
        rdf_run_kwargs=None,
        solute_name="solute",
        analysis_classes=None,
        networking_solvents=None,
        verbose=False,
    ):
        super(Solution, self).__init__(solute.universe.trajectory, verbose=verbose)
        self.radii = radii or {}
        self.solvent_counts = solvent_counts or {}
        for name in solvents.keys():
            if name not in self.solvent_counts.keys():
                self.solvent_counts[name] = len(solvents[name].residues)
        self.kernel = rdf_kernel or identify_cutoff_scipy
        self.kernel_kwargs = kernel_kwargs or {}
        self.rdf_init_kwargs = rdf_init_kwargs or {}
        self.rdf_run_kwargs = rdf_run_kwargs or {}
        self.has_run = False
        self.u = solute.universe
        self.solute = solute
        self.n_solute = len(self.solute.residues)
        self.solute_res_ix = solute.residues.ix
        self.solute_atom_ix = solute.atoms.ix
        self.solvents = solvents
        self.solute_name = solute_name
        self.res_name_map = pd.Series(['none'] * len(self.u.residues))
        self.res_name_map[self.solute.residues.ix] = self.solute_name
        for name, solvent in solvents.items():
            self.res_name_map[solvent.residues.ix] = name
        # logic for instantiating analysis classes.
        if analysis_classes is None:
            self.analysis_classes = ["pairing", "coordination", "speciation"]
        else:
            self.analysis_classes = [cls.lower() for cls in analysis_classes]
        if "networking" in self.analysis_classes and networking_solvents is None:
            raise ValueError(
                "networking analysis requires networking_solvents to be provided."
            )
        else:
            self.networking_solvents = networking_solvents

    def _prepare(self):
        """
        This function identifies the solvation radii and saves the associated RDF data.
        """
        self.rdf_data = {}
        self.solvation_data = None
        self.solvation_data_dup = None
        self.speciation = None
        self.pairing = None
        self.coordination = None
        self.solvation_frames = []
        assert self.u.trajectory.dimensions is not None, (
            "Dimensions of the trajectory must be set to compute an rdf. Consider using "
            "MDAnalysis.transformations.set_dimensions."
        )
        for name, solvent in self.solvents.items():
            # generate and save RDFs
            self.rdf_init_kwargs["range"] = self.rdf_init_kwargs.get("range") or (0, 7.5)
            rdf = InterRDF(self.solute, solvent, **self.rdf_init_kwargs)
            self.rdf_run_kwargs["stop"] = self.rdf_run_kwargs.get("stop") or self.stop
            self.rdf_run_kwargs["step"] = self.rdf_run_kwargs.get("step") or self.step
            self.rdf_run_kwargs["start"] = self.rdf_run_kwargs.get("start") or self.start
            rdf.run(**self.rdf_run_kwargs)
            bins, data = rdf.results.bins, rdf.results.rdf
            self.rdf_data[name] = (bins, data)
            # generate and save plots
            if name not in self.radii.keys():
                self.radii[name] = self.kernel(bins, data, **self.kernel_kwargs)
        calculated_radii = set([name for name, radius in self.radii.items()
                                if not np.isnan(radius)])
        missing_solvents = set(self.solvents.keys()) - calculated_radii
        missing_solvents_str = ' '.join([str(i) for i in missing_solvents])
        assert len(missing_solvents) == 0, (
            f"Solution could not identify a solvation radius for "
            f"{missing_solvents_str}. Please manually enter missing radii "
            f"by editing the radii dict and rerun the analysis."
        )

    def _single_frame(self):
        """
        This function finds the solvation shells of each solute at a given frame.
        """
        # initialize empty lists
        pairs_list = []
        dist_list = []
        tags_list = []
        # loop to find solvated atoms of each type
        for name, solvent in self.solvents.items():
            pairs, dist = capped_distance(
                self.solute.positions,
                solvent.positions,
                self.radii[name],
                box=self.u.dimensions,
            )
            # replace local ids with absolute ids
            pairs[:, 1] = solvent.ix[[pairs[:, 1]]]
            # extend
            pairs_list.append(pairs)
            dist_list.append(dist)
            tags_list.append(np.full(len(dist), name))  # creating a name array
        # create full length features arrays
        pairs_array = np.concatenate(pairs_list, dtype=int)
        dist_array = np.concatenate(dist_list)
        res_name_array = np.concatenate(tags_list)
        res_ix_array = self.u.atoms[pairs_array[:, 1]].resindices
        array_length = len(pairs_array)
        frame_number_array = np.full(array_length, self._ts.frame)
        # stack the data into one large array
        solvation_data_np = np.column_stack(
            (frame_number_array, pairs_array[:, 0], pairs_array[:, 1], dist_array, res_name_array, res_ix_array)
        )
        # add the current frame to the growing list of solvation arrays
        self.solvation_frames.append(solvation_data_np)

    def _conclude(self):
        """
        Creates a clean solvation_data pandas.DataFrame and instantiates several analysis classes.
        """
        # stack all solvation frames into a single data structure
        solvation_data_np = np.vstack(self.solvation_frames)
        solvation_data_df = pd.DataFrame(
            solvation_data_np,
            # TODO: replace solvated_atom with solute?
            columns=["frame", "solvated_atom", "atom_ix", "dist", "res_name", "res_ix"]
        )
        # clean up solvation_data df
        for column in ["frame", "solvated_atom", "atom_ix", "dist", "res_ix"]:
            solvation_data_df[column] = pd.to_numeric(solvation_data_df[column])
        solvation_data_dup = solvation_data_df.sort_values(["frame", "solvated_atom", "dist"])
        solvation_data = solvation_data_dup.drop_duplicates(["frame", "solvated_atom", "res_ix"])
        self.solvation_data_dup = solvation_data_dup.set_index(["frame", "solvated_atom", "atom_ix"])
        self.solvation_data = solvation_data.set_index(["frame", "solvated_atom", "atom_ix"])
        # instantiate analysis classes
        self.has_run = True
        classes_dict = {
            'speciation': Speciation,
            'pairing': Pairing,
            'coordination': Coordination,
            'residence': Residence,
            'networking': Networking,
        }
        for analysis_class in self.analysis_classes:
            if analysis_class == 'networking':
                setattr(self, 'networking', Networking.from_solution(self, self.networking_solvents))
            else:
                setattr(self, analysis_class, classes_dict[analysis_class].from_solution(self))

    @staticmethod
    def _plot_solvation_radius(bins, data, radius):
        """
        Plot a solvation radius on an RDF.

        Includes a vertical line at the radius of interest.

        Parameters
        ----------
        bins : np.array
            the RDF bins
        data : np.array
            the RDF data
        radius : float
            the cutoff radius to draw on the plot

        Returns
        -------
        fig : matplotlib.Figure
        ax : matplotlib.Axes
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
        Plot the RDF of a solvent molecule

        Specified by the residue name in the solvents dict. Includes a vertical
        line at the radius of interest.

        Parameters
        ----------
        res_name : str
            the name of the residue of interest, as written in the solvents dict

        Returns
        -------
        fig : matplotlib.Figure
        ax : matplotlib.Axes
        """
        bins, data = self.rdf_data[res_name]
        fig, ax = self._plot_solvation_radius(bins, data, self.radii[res_name])
        ax.set_title(f"Solvation distance of {res_name}")
        return fig, ax

    def radial_shell(self, solute_index, radius):
        """
        Select all residues with atoms within r of the solute.

        The solute is specified by it's index within solvation_data. r is
        specified with the radius argument. Thin wrapper around
        solvation.get_radial_shell.

        Parameters
        ----------
        solute_index : int
            the index of the solute of interest
        radius : float or int
            radius used for atom selection

        Returns
        -------
        MDAnalysis.AtomGroup
        """
        return get_radial_shell(self.solute[solute_index], radius)

    def closest_n_mol(self, solute_index, n_mol, **kwargs):
        """
        Select the n closest mols to the solute.

        The solute is specified by it's index within solvation_data.
        n is specified with the n_mol argument. Optionally returns
        an array of their resids and an array of the distance of
        the closest atom in each molecule. Thin wrapper around
        solvation.get_closest_n_mol, see documentation for more detail.

        Parameters
        ----------
        solute_index : int
            The index of the solute of interest
        n_mol : int
            The number of molecules to return
        kwargs : passed to solvation.get_closest_n_mol

        Returns
        -------
        full shell : MDAnalysis.AtomGroup
            the atoms in the shell
        ordered_resids : numpy.array of int, optional
            the residue id of the n_mol closest atoms
        radii : numpy.array of float, optional
            the distance of each atom from the center
        """
        return get_closest_n_mol(self.solute[solute_index], n_mol, **kwargs)

    def solvation_shell(self, solute_index, frame, as_df=False, remove_mols=None, closest_n_only=None):
        """
        Select the solvation shell of the solute.

        The solvation shell can be returned either as an
        AtomGroup, to be visualized or passed to other routines,
        or as a pandas.DataFrame for convenient inspection.

        The solvation shell can be truncated before being returned,
        either by removing specific residue types with remove_mols
        or by introducing a hard cutoff with closest_n_only.

        Parameters
        ----------
        solute_index : int
            The index of the solute of interest
        frame : int
            the frame in the trajectory to perform selection at. Defaults to the
            current trajectory frame.
        as_df : bool, default False
            if true, this function will return a DataFrame representing the shell
            instead of a AtomGroup.
        remove_mols : dict of {str: int}, optional
            remove_dict lets you remove specific residues from the final shell.
            It should be a dict of molnames and ints e.g. ``{'mol1': n, 'mol2', m}``.
            It will remove up to n of mol1 and up to m of mol2. So if the dict is
            ``{'mol1': 1, 'mol2', 1}`` and the shell has 4 mol1 and 0 mol2,
            solvation_shell will return a shell with 3 mol1 and 0 mol2.
        closest_n_only : int, optional
            if given, only the closest n residues will be included

        Returns
        -------
        MDAnalysis.AtomGroup or pandas.DataFrame

        """
        assert self.has_run, "Solute.run() must be called first."
        assert frame in self.frames, ("The requested frame must be one "
                                      "of an analyzed frames in self.frames.")
        remove_mols = {} if remove_mols is None else remove_mols
        # select shell of interest
        shell = self.solvation_data.xs((frame, solute_index), level=("frame", "solvated_atom"))
        # remove mols
        for mol_name, n_remove in remove_mols.items():
            # first, filter for only mols of type mol_name
            is_mol = shell.res_name == mol_name
            res_ix = shell[is_mol].res_ix
            mol_count = len(res_ix)
            n_remove = min(mol_count, n_remove)
            # then truncate resnames to remove mols
            remove_ix = res_ix[(mol_count - n_remove):]
            # then apply to original shell
            remove = shell.res_ix.isin(remove_ix)
            shell = shell[np.invert(remove)]
        # filter based on length
        if closest_n_only:
            assert closest_n_only > 0, "closest_n_only must be at least 1"
            closest_n_only = min(len(shell), closest_n_only)
            shell = shell[0: closest_n_only]
        if as_df:
            return shell
        else:
            return self._df_to_atom_group(shell, solute_index=solute_index)

    def _df_to_atom_group(self, df, solute_index=None):
        """
        Selects an MDAnalysis.AtomGroup from a pandas.DataFrame with res_ix.

        Parameters
        ----------
        df : pandas.DataFrame
            a df with a 'res_ix' column
        solute_index : int, optional
            if given, will include the solute with solute_index

        Returns
        -------
        MDAnalysis.AtomGroup
        """
        ix = df['res_ix'].values  # -1 to go from res_ix -> res_ix
        atoms = self.u.residues[ix].atoms
        if solute_index is not None:
            atoms = atoms | self.solute[solute_index]
        return atoms
