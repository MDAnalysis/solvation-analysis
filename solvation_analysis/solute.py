"""
========
Solute
========
:Author: Orion Cohen
:Year: 2021
:Copyright: GNU Public License v3

Solute is the core of solvation_analysis. Once you have specified the
solute and solvent, Solute can calculate the coordination, networking,
pairing, speciation, and residence information for every solute in
the system.

If you are just getting started with solvation_analysis, the
tutorials are the best place to start. If you are looking to learn
more, this page will provide a full API overview of the Solute class
and go into more detail on it's inner working. There are three core
concepts to understand in the Solute class: solute_atoms vs solvents,
solvation data DataFrame, and the ``atom_solutes`` dictionary.

**solute vs solvents**

To instantiate a Solute, one must first use ``MDAnalysis`` to select
the solute_atoms and solvents. The solute_atoms can be organized
in different ways (described below) but the solvents are always
a dictionary of solvent names (strings) and solvent AtomGroups. In
some cases, specifying the solvent and solutes is the hardest part.
Note that one species might be both the solute and a solvent. Such
as if you are interested in hydrogen bond networks in water.

**the solvation data DataFrame**

The solvation data DataFrame is the core of the Solute class. It contains
the full solvation shell of every solute in the system at every timestep.
It is a labeled adjacency matrix of the solvation network. From this, all
other solvation information is derived: coordination numbers, residence
times, solute-solvent pairing, etc. The DataFrame does not need to be
accessed by the user, but it is at the core of how solvation_analysis works.

**the atom_solutes dictionary**

Solvation can be considered on two scales, the scale of the solute molecule
or the scale of atoms in the solute. For example, one might be interested in
the solvation shell around a whole solute molecule, or around just a specific
functional group. The ``Solute`` allows both scales to be analyzed simultaneously.
It first constructs a ``Solute`` for every unique atom in the `solute_atoms`
AtomGroup and then combines them into a single ``Solute`` representing the whole
molecule. The ``Solute`` for each atom is stored in the ``atom_solutes`` dictionary.
The key's of the dictionary are the ``solute_name`` of each ``Solute``, and
depending on on how the ``Solute`` was instantiated, these might be generic names
(e.g. "solute_0", "solute_1", etc.) or user-specified.


A Solute can be instantiated with the ``from_atoms``, ``from_atoms_dict``, or
``from_solutes_list`` methods. Below, the basic use of each constructor is
demonstrated.

``from_atoms`` takes an ``MDAnalysis.AtomGroup`` as input and
automatically parses it into individual ``atom_solutes``. This is the
least flexible constructor and does the most work behind the scenes.

.. code-block:: python

    water = u.select_atoms("resname water")
    water_O = water.select_atoms("element O")
    water_O_solute = Solute.from_atoms(water_O, {"water": water})
    # OR
    water_solute = Solute.from_atoms(water, {"water": water})

The `from_atoms_dict` constructor takes a dict of solute names (strings)
and `AtomGroups`. Each `AtomGroup` must have only have one atom per
solute. This provides an intermediate level of flexibility because
it allows you to name your atom solutes.

.. code-block:: python

    water_solute = Solute.from_atoms_dict({"water_O": water_O, "water_H1": water_H1, "water_H2": water_H2}, {"water": water)

Finally, the `from_solutes_list` is the most flexible because
it allows you to create all the solutes individually with
different settings. It might be useful if you wanted to use
specify radii for each ``solute_atom``.

.. code-block:: python

    water_H1 = u.select_atoms(...)
    water_H2 = u.select_atoms(...)
    solute_O = Solute(water_O, {"water": water})
    solute_H1 = Solute(water_H1, {"water": water})
    solute_H2 = Solute(water_H2, {"water": water})

    water_solute = Solute.from_solutes_list([solute_O, solute_H1, solute_H2])

Once the ``Solute`` is instantiated, it must be run by calling ``Solute.run()``.
This will calculate the solvation data and instantiate the analysis classes. At
this point, the ``Solute`` is ready to be analyzed. Solute inherits from
MDAnalysis.analysis.base.AnalysisBase, so ``run`` supports all arguments
from the base class.

By default, ``Coordination``, ``Pairing``, and ``Speciation`` are instantiated
as attributes of the ``Solute``. ``Networking`` and ``Residence`` can also be
added. These are covered in greater detail in their respective API pages.

Solute also provides several functions to select a particular solute and its
solvation shell, returning an AtomGroup for visualization or further analysis.
This is covered in the visualization tutorial.
"""
from collections import defaultdict
from functools import reduce

import matplotlib.pyplot as plt
import pandas as pd
import warnings

import MDAnalysis as mda
from MDAnalysis.analysis.base import AnalysisBase
from MDAnalysis.analysis.rdf import InterRDF
from MDAnalysis.lib.distances import capped_distance
import numpy as np

from solvation_analysis._utils import verify_solute_atoms, verify_solute_atoms_dict, get_closest_n_mol, get_radial_shell
from solvation_analysis.rdf_parser import identify_cutoff_scipy
from solvation_analysis.coordination import Coordination
from solvation_analysis.networking import Networking
from solvation_analysis.pairing import Pairing
from solvation_analysis.residence import Residence
from solvation_analysis.speciation import Speciation

from solvation_analysis._column_names import *


class Solute(AnalysisBase):
    """
    Analyze the solvation structure of a solute.

    Solute determines the coordination between the solute and solvents,
    collecting that data in a pandas.DataFrame (solvation_data) for further
    analysis. The solvation data DataFrame is a labeled adjacency matrix of
    the first-shell solvation network.

    By default, Solute will determine the coordination cutoff
    radii from the Solute-solvent RDF. Radii can also be supplied
    with the radii parameter. The RDF's of each solute-solvent
    pari can be visualize with the plot_solvation_radius method.

    Solute instantiates Speciation, Coordination, and Pairing
    objects from the solvation_data, providing a convenient interface for
    further analysis. Networking and Residence can be added by modifying
    the ``analysis_classes`` parameter.

    The parameters below are the ``kwargs`` supplied to the ``from_*``
    constructors.

    Parameters
    ----------
    radii : dict of {str: float}, optional
        an optional dictionary of solvent names and associated solvation radii
        e.g. {"name_2": radius_2, "name_5": radius_5} Any radii not given will
        be calculated. The solvent names should match the solvents parameter.
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
        stop will use any kwargs provided to ``solute.run(**kwargs)``.
    solute_name: str, optional
        the name of the solute, used for labeling.
    analysis_classes : List[str] or str, optional
        a list of the analysis classes to be instantiated, current options are:
        "pairing", "coordination", "speciation", "residence", and "networking".
        By default, only "pairing", "coordination", and "residence" are instantiated.
        If networking is included, the networking_solvents kwarg must be specified.
        Enter 'all' to include all analysis classes.
    networking_solvents : str, optional
        see the `solvents` parameter of the Networking class.
    verbose : bool, optional
       Turn on more logging and debugging, default ``False``

    Attributes
    ----------
    u : Universe
        An MDAnalysis.Universe object
    atom_solutes : dict of {str: Solute}
        a dictionary of solute names and their associated Solute objects.
    n_solutes : int
        number of solute atoms
    radii : dict
        a dictionary of solvation radii for each solvent
        e.g. {"name_2": radius_2, "name_2": radius_2, ...}
    rdf_data : dict
        a dictionary of RDF data, keys are solvent names and values
        are (bins, data) tuples.
    solvation_data : pandas.DataFrame
        a dataframe of solvation data with columns "frame", "solute_atom", "solvent_atom",
        "distance", "solvent_name", and "solvent". If multiple entries share a frame, solute_atom,
        and solvent_atom, all but the closest atom is dropped.
    solvation_data_dupicates : pandas.DataFrame
        All rows that are dropped from solvation_data when duplicates are dropped.
    solute_res_ix : np.array
        a numpy array of the residue indices of every solute.
    solute_atom_ix : np.array
        a numpy array of the atom indices of every solute.
    solvent_counts : dict of {str: int}
        a dictionary of the number of residues for each solvent.
    res_name_map : pandas.Series
        a map from residue indices in the Universe to solvent and solute names from
        the solute. For example, if the first residue in the universe was in
        ``self.solvent['res_one']``, then ``res_name_map[0] == 'res_one'``.
    pairing : pairing.Pairing (optional)
        pairing analyzes the fraction solutes and solvents that are coordinated.
    coordination : coordination.Coordination (optional)
        coordination analyses the coordination numbers of solvents and which
        solvent atoms are coordinated.
    speciation : speciation.Speciation (optional)
        speciation provides an interface for finding and selecting the solvation shells
        surrounding each solute.
    residence : residence.Residence (optional)
        residence calculates the residence times of each solvent on the solute. Only
        instantiated if 'residence' is included in the analysis_classes kwarg.
    networking : networking.Networking (optional)
        networking analyses the connectivity of solute-solvent networks. Only instantiated
        if 'networking' is included in the analysis_classes kwarg. the networking_solvents
        kwarg must be specified.
    """

    def __init__(
            self,
            solute_atoms,
            solvents,
            atom_solutes=None,
            radii=None,
            rdf_kernel=None,
            kernel_kwargs=None,
            rdf_init_kwargs=None,
            rdf_run_kwargs=None,
            solute_name="solute_0",
            analysis_classes=None,
            networking_solvents=None,
            verbose=False,
            internal_call=False,
    ):
        """
        This method is not intended to be called directly. Instead, use
        ``from_atoms``, ``from_atoms_dict`` or ``from_solute_list`` to create a Solute.
        """
        if not internal_call:
            raise RuntimeError("Please use Solute.from_atoms, Solute.from_atoms_dict, or "
                               "Solute.from_solute_list instead of the default constructor.")
        super(Solute, self).__init__(solute_atoms.universe.trajectory, verbose=verbose)

        self.solute_atoms = solute_atoms  # TODO: this shit!
        self.atom_solutes = atom_solutes
        if self.atom_solutes is None or len(atom_solutes) <= 1:
            self.atom_solutes = {solute_name: self}
        self.radii = radii or {}
        self.solvent_counts = {name: atoms.n_residues for name, atoms in solvents.items()}
        self.kernel = rdf_kernel or identify_cutoff_scipy
        self.kernel_kwargs = kernel_kwargs or {}
        self.rdf_init_kwargs = rdf_init_kwargs or {}
        self.rdf_run_kwargs = rdf_run_kwargs or {}
        self.has_run = False
        self.u = solute_atoms.universe
        self.n_solutes = solute_atoms.n_residues
        self.solute_res_ix = pd.Series(solute_atoms.atoms.resindices, solute_atoms.atoms.ix)
        self.solute_name = solute_name
        self.solvents = solvents

        # instantiate the res_name_map
        self.res_name_map = pd.Series(['none'] * len(self.u.residues))
        self.res_name_map[solute_atoms.residues.ix] = self.solute_name
        for name, solvent in solvents.items():
            self.res_name_map[solvent.residues.ix] = name

        # instantiate analysis classes.
        if analysis_classes is None:
            self.analysis_classes = ["pairing", "coordination", "speciation"]
        elif analysis_classes == 'all':
            self.analysis_classes = ["pairing", "coordination", "speciation", "residence", "networking"]
        else:
            self.analysis_classes = [cls.lower() for cls in analysis_classes]
        if "networking" in self.analysis_classes and networking_solvents is None:
            raise ValueError(
                "networking analysis requires networking_solvents to be provided."
            )
        else:
            self.networking_solvents = networking_solvents

    @staticmethod
    def from_atoms(solute_atoms, solvents, rename_solutes=None, **kwargs):
        """
        Create a Solute from a single AtomGroup. The solute_atoms AtomGroup must
        should contain identical residues and identical atoms on each residue.
        For example, if one wanted to look at the O and H atoms of an ethanol molecule,
        solute_atoms should contain the O and H atoms of all ethanol molecules. It
        could not contain C atoms from some ethanol or include residues
        that were not ethanol.

        Parameters
        ----------
        solute_atoms : AtomGroup
            an AtomGroup or ResidueGroup containing one atom per residue.
        solvents: dict of {str: MDAnalysis.AtomGroup}
            a dictionary of solvent names and associated MDAnalysis.AtomGroups.
            e.g. {"name_1": solvent_group_1,"name_2": solvent_group_2, ...}
        rename_solutes : dict, optional
            a dictionary of solute names to rename the solutes. Keys are the
            default solute_name, and values are the new names.
            For example, from_atoms might return a solute with solute_name
            "solute_0", but you want to rename it to "functional_group_X".
            In this case, rename_solutes={"solute_0": "functional_group_X"}.
            If None, solutes will be numbered in the order their atoms appear
            on the residue.
        kwargs: dict
            All kwargs listed in the Parameters section of the Solute class.

        Returns
        -------
        Solute

        """
        rename_solutes = rename_solutes or {}
        # a dict with keys as integers and values as AtomGroups
        # this will get called again later on with the same input, but for now, it's fine
        solute_atom_group_dict = verify_solute_atoms(solute_atoms)
        solute_atom_group_dict_renamed = {
            rename_solutes.get(f"solute_{i}") or f"solute_{i}": atom_group
            for i, atom_group in solute_atom_group_dict.items()
        }
        return Solute.from_atoms_dict(solute_atom_group_dict_renamed, solvents, **kwargs)

    @staticmethod
    def from_atoms_dict(solute_atoms_dict, solvents, **kwargs):
        """
        Create a Solute object from a dictionary of solute atoms.

        Parameters
        ----------
        solute_atoms_dict: dict of {str: MDAnalysis.AtomGroup}
            a dictionary of solute atoms, e.g. {"name_1": solute_1, "name_2": solute_2}.
            Each AtomGroup should contain one type of atom, and only one atom per
            residue.
        solvents: dict of {str: MDAnalysis.AtomGroup}
            a dictionary of solvent names and associated MDAnalysis.AtomGroups.
            e.g. {"name_1": solvent_group_1,"name_2": solvent_group_2, ...}        kwargs
        kwargs: dict
            All kwargs listed in the Parameters section of the Solute class.

        Returns
        -------
        Solute

        """
        # all solute AtomGroups in one AtomGroup + verification
        assert isinstance(solute_atoms_dict, dict), ("Solute_atoms_dict must be a dict.")
        solute_atom_group = verify_solute_atoms_dict(solute_atoms_dict)

        # create the solutes for each atom
        atom_solutes = {}
        for solute_name, atoms in solute_atoms_dict.items():
            atom_solutes[solute_name] = Solute(
                atoms,
                solvents,
                internal_call=True,
                **{**kwargs, "solute_name": solute_name}
            )
        # create the solute for the whole solute
        solute = Solute(
            solute_atom_group,
            solvents,
            atom_solutes=atom_solutes,
            internal_call=True,
            **kwargs
        )
        if len(atom_solutes) > 1:
           solute.run = solute._run_solute_atoms
        return solute

    @staticmethod
    def from_solute_list(solutes, solvents, **kwargs):
        """
        Create a Solute from a list of Solutes. All Solutes must have only a
        single solute atom on each solute residue. Essentially, from_solute_list
        allows you to preconstruct the ``atom_solutes`` dictionary. This is useful
        if you want to individually specify initialization parameters for each
        atom Solute, otherwise, a different constructor is recommended.

        Parameters
        ----------
        solutes: list of Solute
            A list of Solutes. All Solutes must have only a single solute atom.
        on each solute residue.
        solvents: dict of {str: MDAnalysis.AtomGroup}
            a dictionary of solvent names and associated MDAnalysis.AtomGroups.
            e.g. {"name_1": solvent_group_1,"name_2": solvent_group_2, ...}
        kwargs: dict
            All kwargs listed in the Parameters section of the Solute class.

        Returns
        -------
        Solute

        """
        # check types and name uniqueness
        for solute in solutes:
            assert type(solute) == Solute, "solutes must be a list of Solute objects."
            assert len(solute.solute_atoms.atoms) == len(solute.solute_atoms.atoms.residues), (
                "Each Solute in solutes must have only a single atom per residue."
            )
        solute_names = [solute.solute_name for solute in solutes]
        assert len(np.unique(solute_names)) == len(solute_names), (
            "The solute_name for each solute must be unique."
        )

        solute_atom_group = reduce(lambda x, y: x | y, [solute.solute_atoms for solute in solutes])
        verify_solute_atoms(solute_atom_group)

        atom_solutes = {solute.solute_name: solute for solute in solutes}
        solute = Solute(
            solute_atom_group,
            solvents,
            atom_solutes=atom_solutes,
            internal_call=True,
            **kwargs
        )
        if len(atom_solutes) > 1:
           solute.run = solute._run_solute_atoms
        return solute

    def _run_solute_atoms(self, start=None, stop=None, step=None, verbose=None):
        # like prepare
        atom_solutes = {}
        rdf_data = {}
        solvation_datas = []
        solvation_data_dups = []
        start, stop, step = self.u.trajectory.check_slice_indices(start, stop, step)
        self.start, self.stop, self.step = start, stop, step
        self.n_frames = len(range(start, stop, step))

        # like run
        for solute in self.atom_solutes.values():
            if not solute.has_run:
                solute.run(start=start, stop=stop, step=step, verbose=verbose)
            if (start, stop, step) != (solute.start, solute.stop, solute.step):
                warnings.warn(f"The start, stop, or step for {solute.solute_name} do not"
                              f"match the start, stop, or step for the run command so it "
                              f"is being re-run.")
                solute.run(start=start, stop=stop, step=step, verbose=verbose)
            if self.solvents != solute.solvents:
                warnings.warn(f"The solvents for {solute.solute_name} do not match the "
                              f"solvents for the run command so it is being re-run.")
                solute.run(start=start, stop=stop, step=step, verbose=verbose)
            atom_solutes[solute.solute_name] = solute
            rdf_data[solute.solute_name] = solute.rdf_data[solute.solute_name]
            solvation_datas.append(solute.solvation_data)
            solvation_data_dups.append(solute.solvation_data_duplicates)

        self.rdf_data = rdf_data
        self.solvation_data = pd.concat(solvation_datas).sort_index()
        self.solvation_data_duplicates = pd.concat(solvation_data_dups)
        self.has_run = True

        # like conclude
        analysis_classes = {
            'speciation': Speciation,
            'pairing': Pairing,
            'coordination': Coordination,
            'residence': Residence,
            'networking': Networking,
        }
        for analysis_class in self.analysis_classes:
            if analysis_class == 'networking':
                setattr(self, 'networking', Networking.from_solute(self, self.networking_solvents))
            else:
                setattr(self, analysis_class, analysis_classes[analysis_class].from_solute(self))

    def _prepare(self):
        """
        This function identifies the solvation radii and saves the associated RDF data.
        """
        self.rdf_data = defaultdict(dict)
        self.solvation_data = None
        self.solvation_data_duplicates = None
        self._solvation_frames = []
        for name, solvent in self.solvents.items():
            # set kwargs with defaults
            self.rdf_init_kwargs["range"] = self.rdf_init_kwargs.get("range") or (0, 7.5)
            self.rdf_init_kwargs["norm"] = self.rdf_init_kwargs.get("norm") or "density"
            self.rdf_run_kwargs["stop"] = self.rdf_run_kwargs.get("stop") or self.stop
            self.rdf_run_kwargs["step"] = self.rdf_run_kwargs.get("step") or self.step
            self.rdf_run_kwargs["start"] = self.rdf_run_kwargs.get("start") or self.start
            # generate and save RDFs
            if self.solute_atoms.intersection(solvent).n_atoms == 0:
                # the solute IS NOT in a solvent, the usual case
                rdf = InterRDF(self.solute_atoms, solvent, **self.rdf_init_kwargs)
                rdf.run(**self.rdf_run_kwargs)
                bins, data = rdf.results.bins, rdf.results.rdf
            else:
                # the solute IS in a solvent
                # we divide the solute and solvent into two groups, so that the rdfs
                # are not contaminated by the solute-solvent pairs.
                halfway_point = self.solute_atoms.n_residues // 2
                solute_half = self.solute_atoms[halfway_point:]
                solvent_half = (solvent.residues - solute_half.residues).atoms
                # this is hacky and will make our rdf noisier but it was easy to implement
                rdf = InterRDF(solute_half, solvent_half, **self.rdf_init_kwargs)
                rdf.run(**self.rdf_run_kwargs)
                bins, data = rdf.results.bins, rdf.results.rdf
            self.rdf_data[self.solute_name][name] = (bins, data)
            # generate and save plots
            if name not in self.radii.keys():
                self.radii[name] = self.kernel(bins, data, **self.kernel_kwargs)
        calculated_radii = set([name for name, radius in self.radii.items()
                                if not np.isnan(radius)])
        missing_solvents = set(self.solvents.keys()) - calculated_radii
        missing_solvents_str = ' '.join([str(i) for i in missing_solvents])
        assert len(missing_solvents) == 0, (
            f"Solute could not identify a solvation radius for "
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
                self.solute_atoms.positions,
                solvent.positions,
                self.radii[name],
                box=self.u.dimensions,
            )
            # make sure pairs don't include intra-molecular interactions
            filter = self.solute_atoms.resindices[pairs[:, 0]] == solvent.resindices[pairs[:, 1]]
            pairs = pairs[~filter]
            dist = dist[~filter]
            # replace local ids with absolute ids
            pairs[:, 0] = self.solute_atoms.atoms.ix[tuple([pairs[:, 0]])]
            pairs[:, 1] = solvent.ix[tuple([pairs[:, 1]])]
            # extend
            pairs_list.append(pairs)
            dist_list.append(dist)
            tags_list.append(np.full(len(dist), name))  # creating a name array
        # create full length features arrays
        pairs_array = np.concatenate(pairs_list, dtype=int)
        frame_number_array = np.full(len(pairs_array), self._ts.frame)
        solute_res_ix_array = self.u.atoms[pairs_array[:, 0]].resindices
        solvent_res_ix_array = self.u.atoms[pairs_array[:, 1]].resindices
        dist_array = np.concatenate(dist_list)
        solute_res_name_array = np.full(len(pairs_array), self.solute_name)
        solvent_res_name_array = np.concatenate(tags_list)
        # stack the data into one large array
        solvation_data_np = np.column_stack(
            (
                frame_number_array,
                solute_res_ix_array,
                pairs_array[:, 0],
                pairs_array[:, 1],
                dist_array,
                solute_res_name_array,
                solvent_res_name_array,
                solvent_res_ix_array
            )
        )
        # add the current frame to the growing list of solvation arrays
        self._solvation_frames.append(solvation_data_np)

    def _conclude(self):
        """
        Creates a clean solvation_data pandas.DataFrame and instantiates several analysis classes.
        """
        # stack all solvation frames into a single data structure
        solvation_data_np = np.vstack(self._solvation_frames)
        solvation_data_df = pd.DataFrame(
            solvation_data_np,
            columns=[
                FRAME,
                SOLUTE_IX,
                SOLUTE_ATOM_IX,
                SOLVENT_ATOM_IX,
                DISTANCE,
                SOLUTE,
                SOLVENT,
                SOLVENT_IX
            ]
        )
        # clean up solvation_data df
        for column in [FRAME, SOLUTE_IX, SOLUTE_ATOM_IX, SOLVENT_ATOM_IX, DISTANCE, SOLVENT_IX]:
            solvation_data_df[column] = pd.to_numeric(solvation_data_df[column])
        solvation_data_df = solvation_data_df.sort_values([FRAME, SOLUTE_ATOM_IX, DISTANCE])
        solvation_data_duplicates = solvation_data_df.duplicated(subset=[FRAME, SOLUTE_ATOM_IX, SOLVENT_IX])
        solvation_data = solvation_data_df[~solvation_data_duplicates]
        self.solvation_data = solvation_data.set_index([FRAME, SOLUTE_IX, SOLUTE_ATOM_IX, SOLVENT_ATOM_IX])
        duplicates = solvation_data_df[solvation_data_duplicates]
        self.solvation_data_duplicates = duplicates.set_index([FRAME, SOLUTE_IX, SOLUTE_ATOM_IX, SOLVENT_ATOM_IX])
        # instantiate analysis classes
        self.has_run = True
        analysis_classes = {
            'speciation': Speciation,
            'pairing': Pairing,
            'coordination': Coordination,
            'residence': Residence,
            'networking': Networking,
        }
        for analysis_class in self.analysis_classes:
            if analysis_class == 'networking':
                setattr(self, 'networking', Networking.from_solute(self, self.networking_solvents))
            else:
                setattr(self, analysis_class, analysis_classes[analysis_class].from_solute(self))

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

    def plot_solvation_radius(self, solute_name, solvent_name):
        """
        Plot the RDF of a solvent molecule

        Specified by the residue name in the solvents dict. Includes a vertical
        line at the radius of interest.

        Parameters
        ----------
        solvent_name : str
            the name of the residue of interest, as written in the solvents dict

        Returns
        -------
        fig : matplotlib.Figure
        ax : matplotlib.Axes
        """
        if len(self.atom_solutes) == 1:
            solute_name = self.solute_name
        assert self.has_run, "Solute.run() must be called first."
        bins, data = self.rdf_data[solute_name][solvent_name]
        radius = self.atom_solutes[solute_name].radii[solvent_name]
        fig, ax = self._plot_solvation_radius(bins, data, radius)
        ax.set_title(f"{self.solute_name} solvation distance for {solvent_name}")
        return fig, ax

    def draw_molecule(self, residue, filename=None):
        """
        Returns

        Parameters
        ----------
        residue: the str name of the solute or any electrolytes or a MDAnalysis.Residue
        filename: If true, writes a file using the RDKit backend.

        Returns
        -------
        RDKit.Chem.rdchem.Mol

        """
        from rdkit.Chem import Draw
        from rdkit.Chem import rdCoordGen
        from rdkit.Chem.Draw.MolDrawing import DrawingOptions
        DrawingOptions.atomLabelFontSize = 100

        if isinstance(residue, str):
            if residue in [self.solute_name, "solute"]:
                mol = self.solute_atoms.residues[0].atoms.convert_to("RDKIT")
                mol_mda_ix = self.solute_atoms.residues[0].atoms.ix
                solute_atoms_ix0 = {solute.solute_atoms.atoms.ix[0]: solute_name
                                    for solute_name, solute in self.atom_solutes.items()}
                for i, atom in enumerate(mol.GetAtoms()):
                    atom_name = solute_atoms_ix0.get(mol_mda_ix[i])
                    label = f"{i}, " + atom_name if atom_name else str(i)
                    atom.SetProp("atomNote", label)
            elif residue in self.solvents.keys():
                mol = self.solvents[residue].residues[0].atoms.convert_to("RDKIT")
                for i, atom in enumerate(mol.GetAtoms()):
                    atom.SetProp("atomNote", str(i))
            else:
                raise ValueError("If the residue is a string, it must be the name of a solute, "
                             "the name of a solvent, or 'solute'.")
        else:
            assert isinstance(residue, mda.core.groups.Residue)
            mol = residue.atoms.convert_to("RDKIT")
            for i, atom in enumerate(mol.GetAtoms()):
                atom.SetProp("atomNote", str(i))
        if filename:
            Draw.MolToFile(mol, filename=filename)
        rdCoordGen.AddCoords(mol)
        return mol

    def get_shell(self, solute_index, frame, as_df=False, remove_mols=None, closest_n_only=None):
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
            get_shell will return a shell with 3 mol1 and 0 mol2.
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
        shell = self.solvation_data.xs((frame, solute_index), level=(FRAME, SOLUTE_IX))
        # remove mols
        for mol_name, n_remove in remove_mols.items():
            # first, filter for only mols of type mol_name
            is_mol = shell[SOLVENT] == mol_name
            res_ix = shell[is_mol][SOLVENT_IX]
            mol_count = len(res_ix)
            n_remove = min(mol_count, n_remove)
            # then truncate resnames to remove mols
            remove_ix = res_ix[(mol_count - n_remove):]
            # then apply to original shell
            remove = shell[SOLVENT_IX].isin(remove_ix)
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

    def get_closest_n_mol(
            self,
            solute_atom_ix,
            n_mol,
            guess_radius=3,
            return_ordered_resix=False,
            return_radii=False,
    ):
        """
        Select the n closest mols to the solute.

        The solute is specified by it's index within solvation_data.
        n is specified with the n_mol argument. Optionally returns
        an array of their resids and an array of the distance of
        the closest atom in each molecule.

        Parameters
        ----------
        solute_atom_ix : int
            The index of the solute of interest
        n_mol : int
            The number of molecules to return
        guess_radius : float or int
            an initial search radius to look for closest n mol
        return_ordered_resix : bool, default False
            whether to return the resix of the closest n
            molecules, ordered by radius
        return_radii : bool, default False
            whether to return the distance of the closest atom of each
            of the n molecules

        Returns
        -------
        full shell : MDAnalysis.AtomGroup
            the atoms in the shell
        ordered_resids : numpy.array of int, optional
            the residue id of the n_mol closest atoms
        radii : numpy.array of float, optional
            the distance of each atom from the center
        """
        return get_closest_n_mol(
            self.u.atoms[solute_atom_ix],
            n_mol,
            guess_radius,
            return_ordered_resix,
            return_radii,
        )

    def radial_shell(self, solute_atom_ix, radius):
        """
        Select all residues with atoms within r of the solute.

        The solute is specified by it's index within solvation_data. r is
        specified with the radius argument. Thin wrapper around
        solvation.get_radial_shell.

        Parameters
        ----------
        solute_atom_ix : int
            the index of the solute of interest
        radius : float or int
            radius used for atom selection

        Returns
        -------
        MDAnalysis.AtomGroup
        """
        return get_radial_shell(self.solute_atoms[solute_atom_ix], radius)

    def _df_to_atom_group(self, df, solute_index=None):
        """
        Selects an MDAnalysis.AtomGroup from a pandas.DataFrame with solvent.

        Parameters
        ----------
        df : pandas.DataFrame
            a df with a "solvent" column
        solute_index : int, optional
            if given, will include the solute with solute_index

        Returns
        -------
        MDAnalysis.AtomGroup
        """
        ix = df[SOLVENT_IX].values
        atoms = self.u.residues[ix].atoms
        if solute_index is not None:
            atoms = atoms | self.u.atoms.residues[solute_index].atoms
        return atoms
