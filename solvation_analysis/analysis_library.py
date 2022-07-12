"""
================
Analysis Library
================
:Author: Orion Cohen, Tingzheng Hou, Kara Fong
:Year: 2021
:Copyright: GNU Public License v3

Analysis library defines a variety of classes that analyze different aspects of solvation.
These classes are all instantiated with the solvation_data generated
from the Solution class.

While the classes in ``analysis_library`` can be used in isolation, they are meant to be used
as attributes of the Solution class. This makes instantiating them and calculating the
solvation data a non-issue.
"""
import collections
import math
import warnings

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import acovf
from scipy.optimize import curve_fit
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components


class Speciation:
    """
    Calculate the solvation shells of every solute.

    Speciation organizes the solvation data by the type of residue
    coordinated with the central solvent. It collects this information in a
    pandas.DataFrame indexed by the frame and solute number. Each column is
    one of the solvents in the res_name column of the solvation data. The
    column value is how many residue of that type are in the solvation shell.

    Speciation provides the speciation of each solute in the speciation
    attribute, it also calculates the percentage of each unique
    shell and makes it available in the speciation_percent attribute.

    Additionally, there are methods for finding solvation shells of
    interest and computing how common certain shell configurations are.

    Parameters
    ----------
    solvation_data : pandas.DataFrame
        The solvation data frame output by Solution.
    n_frames : int
        The number of frames in solvation_data.
    n_solutes : int
        The number of solutes in solvation_data.

    Attributes
    ----------
    speciation : pandas.DataFrame
        a dataframe containing the speciation of every li ion at
        every trajectory frame. Indexed by frame and solute numbers.
        Columns are the solvent molecules and values are the number
        of solvent in the shell.
    speciation_percent : pandas.DataFrame
        the percentage of shells of each type. Columns are the solvent
        molecules and and values are the number of solvent in the shell.
        The final column is the percentage of total shell of that
        particular composition.
    co_occurrence : pandas.DataFrame
        The actual co-occurrence of solvents divided by the expected co-occurrence.
        In other words, given one molecule of solvent i in the shell, what is the
        probability of finding a solvent j relative to choosing a solvent at random
        from the pool of all coordinated solvents. This matrix will
        likely not be symmetric.
    """

    def __init__(self, solvation_data, n_frames, n_solutes):
        self.solvation_data = solvation_data
        self.n_frames = n_frames
        self.n_solutes = n_solutes
        self.speciation_data, self.speciation_percent = self._compute_speciation()
        self.co_occurrence = self._solvent_co_occurrence()

    @staticmethod
    def from_solution(solution):
        """
        Generate a Speciation object from a solution.

        Parameters
        ----------
        solution : Solution

        Returns
        -------
        Pairing
        """
        assert solution.has_run, "The solution must be run before calling from_solution"
        return Speciation(
            solution.solvation_data,
            solution.n_frames,
            solution.n_solute,
        )

    def _compute_speciation(self):
        counts = self.solvation_data.groupby(["frame", "solvated_atom", "res_name"]).count()["res_ix"]
        counts_re = counts.reset_index(["res_name"])
        speciation_data = counts_re.pivot(columns=["res_name"]).fillna(0).astype(int)
        res_names = speciation_data.columns.levels[1]
        speciation_data.columns = res_names
        sum_series = speciation_data.groupby(speciation_data.columns.to_list()).size()
        sum_sorted = sum_series.sort_values(ascending=False)
        speciation_percent = sum_sorted.reset_index().rename(columns={0: 'count'})
        speciation_percent['count'] = speciation_percent['count'] / (self.n_frames * self.n_solutes)
        return speciation_data, speciation_percent

    @classmethod
    def _mean_speciation(cls, speciation_frames, solute_number, frame_number):
        means = speciation_frames.sum(axis=1) / (solute_number * frame_number)
        return means

    def shell_percent(self, shell_dict):
        """
        Calculate the percentage of shells matching shell_dict.

        This function computes the percent of solvation shells that exist with a particular
        composition. The composition is specified by the shell_dict. The percent
        will be of all shells that match that specification.

        Attributes
        ----------
        shell_dict : dict of {str: int}
            a specification for a shell composition. Keys are residue names (str)
            and values are the number of desired residues. e.g. if shell_dict =
            {'mol1': 4} then the function will return the percentage of shells
            that have 4 mol1. Note that this may include shells with 4 mol1 and
            any number of other solvents. To specify a shell with 4 mol1 and nothing
            else, enter a dict such as {'mol1': 4, 'mol2': 0, 'mol3': 0}.

        Returns
        -------
        float
            the percentage of shells

        Examples
        --------

         .. code-block:: python

            # first define Li, BN, and FEC AtomGroups
            >>> solution = Solution(Li, {'BN': BN, 'FEC': FEC, 'PF6': PF6})
            >>> solution.run()
            >>> solution.speciation.shell_percent({'BN': 4, 'PF6': 1})
            0.0898
        """
        query_list = [f"{name} == {str(count)}" for name, count in shell_dict.items()]
        query = " and ".join(query_list)
        query_counts = self.speciation_percent.query(query)
        return query_counts['count'].sum()

    def find_shells(self, shell_dict):
        """
        Find all solvation shells that match shell_dict.

        This returns the frame, solute index, and composition of all solutes
        that match the composition given in shell_dict.

        Attributes
        ----------
        shell_dict : dict of {str: int}
            a specification for a shell composition. Keys are residue names (str)
            and values are the number of desired residues. e.g. if shell_dict =
            {'mol1': 4} then the function will return all shells
            that have 4 mol1. Note that this may include shells with 4 mol1 and
            any number of other solvents. To specify a shell with 4 mol1 and nothing
            else, enter a dict such as {'mol1': 4, 'mol2': 0, 'mol3': 0}.

        Returns
        -------
        pandas.DataFrame
            the index and composition of all shells that match shell_dict
        """
        query_list = [f"{name} == {str(count)}" for name, count in shell_dict.items()]
        query = " and ".join(query_list)
        query_counts = self.speciation_data.query(query)
        return query_counts

    def _solvent_co_occurrence(self):
        # calculate the co-occurrence of solvent molecules.
        expected_solvents_list = []
        actual_solvents_list = []
        for solvent in self.speciation_data.columns.values:
            # calculate number of available coordinating solvent slots
            shells_w_solvent = self.speciation_data.query(f'{solvent} > 0')
            n_solvents = shells_w_solvent.sum()
            # calculate expected number of coordinating solvents
            n_coordination_slots = n_solvents.sum() - len(shells_w_solvent)
            coordination_percentage = self.speciation_data.sum() / self.speciation_data.sum().sum()
            expected_solvents = coordination_percentage * n_coordination_slots
            # calculate actual number of coordinating solvents
            actual_solvents = n_solvents.copy()
            actual_solvents[solvent] = actual_solvents[solvent] - len(shells_w_solvent)
            # name series and append to list
            expected_solvents.name = solvent
            actual_solvents.name = solvent
            expected_solvents_list.append(expected_solvents)
            actual_solvents_list.append(actual_solvents)
        # make DataFrames
        actual_df = pd.concat(actual_solvents_list, axis=1)
        expected_df = pd.concat(expected_solvents_list, axis=1)
        # calculate correlation matrix
        correlation = actual_df / expected_df
        return correlation

    def plot_co_occurrence(self):
        """
        Plot the co-occurrence matrix of the solution.

        Co-occurrence as a heatmap with numerical values in addition to colors.

        Returns
        -------
        fig : matplotlib.Figure
        ax : matplotlib.Axes

        """
        solvent_names = self.speciation_data.columns.values
        fig, ax = plt.subplots()
        im = ax.imshow(self.co_occurrence)
        # We want to show all ticks...
        ax.set_xticks(np.arange(len(solvent_names)))
        ax.set_yticks(np.arange(len(solvent_names)))
        # ... and label them with the respective list entries
        ax.set_xticklabels(solvent_names, fontsize=14)
        ax.set_yticklabels(solvent_names, fontsize=14)
        # Let the horizontal axes labeling appear on top.
        ax.tick_params(top=True, bottom=False,
                       labeltop=True, labelbottom=False, )
        # Rotate the tick labels and set their alignment.
        plt.setp(ax.get_xticklabels(), rotation=-30, ha="right",
                 rotation_mode="anchor")
        # Loop over data dimensions and create text annotations.
        for i in range(len(solvent_names)):
            for j in range(len(solvent_names)):
                ax.text(j, i, round(self.co_occurrence.iloc[i, j], 2),
                        horizontalalignment="center",
                        verticalalignment="center",
                        color="black",
                        fontsize=14,
                        )
        fig.tight_layout()
        return fig, ax


class Coordination:
    """
    Calculate the coordination number for each solvent.

    Coordination calculates the coordination number by averaging the number of
    coordinated solvents in all of the solvation shells. This is equivalent to
    the typical method of integrating the solute-solvent RDF up to the solvation
    radius cutoff. As a result, Coordination calculates **species-species** coordination
    numbers, not the total coordination number of the solute. So if the coordination
    number of mol1 is 3.2, there are on average 3.2 mol1 residues within the solvation
    distance of each solute.

    The coordination numbers are made available as an mean over the whole
    simulation and by frame.

    Parameters
    ----------
    solvation_data : pandas.DataFrame
        The solvation data frame output by Solution.
    n_frames : int
        The number of frames in solvation_data.
    n_solutes : int
        The number of solutes in solvation_data.

    Attributes
    ----------
    cn_dict : dict of {str: float}
        a dictionary where keys are residue names (str) and values are the
        mean coordination number of that residue (float).
    cn_by_frame : pd.DataFrame
        a DataFrame of the mean coordination number of in each frame of the trajectory.
    coordinating_atoms : pd.DataFrame
        percent of each atom_type participating in solvation, calculated for each solvent.

    Examples
    --------

     .. code-block:: python

        # first define Li, BN, and FEC AtomGroups
        >>> solution = Solution(Li, {'BN': BN, 'FEC': FEC, 'PF6': PF6})
        >>> solution.run()
        >>> solution.coordination.cn_dict
        {'BN': 4.328, 'FEC': 0.253, 'PF6': 0.128}

    """

    def __init__(self, solvation_data, n_frames, n_solutes, atom_group):
        self.solvation_data = solvation_data
        self.n_frames = n_frames
        self.n_solutes = n_solutes
        self.cn_dict, self.cn_by_frame = self._mean_cn()
        self.atom_group = atom_group
        self.coordinating_atoms = self._calculate_coordinating_atoms()

    @staticmethod
    def from_solution(solution):
        """
        Generate a Coordination object from a solution.

        Parameters
        ----------
        solution : Solution

        Returns
        -------
        Pairing
        """
        assert solution.has_run, "The solution must be run before calling from_solution"
        return Coordination(
            solution.solvation_data,
            solution.n_frames,
            solution.n_solute,
            solution.u.atoms,
        )

    def _mean_cn(self):
        counts = self.solvation_data.groupby(["frame", "solvated_atom", "res_name"]).count()["res_ix"]
        cn_series = counts.groupby(["res_name", "frame"]).sum() / (
                self.n_solutes * self.n_frames
        )
        cn_by_frame = cn_series.unstack()
        cn_dict = cn_series.groupby(["res_name"]).sum().to_dict()
        return cn_dict, cn_by_frame

    def _calculate_coordinating_atoms(self, tol=0.005):
        """
        Determine which atom types are actually coordinating
        return the types of those atoms
        """
        # lookup atom types
        atom_types = self.solvation_data.reset_index(['atom_ix'])
        atom_types['atom_type'] = self.atom_group[atom_types['atom_ix']].types
        # count atom types
        atoms_by_type = atom_types[['atom_type', 'res_name', 'atom_ix']]
        type_counts = atoms_by_type.groupby(['res_name', 'atom_type']).count()
        solvent_counts = type_counts.groupby(['res_name']).sum()['atom_ix']
        # calculate percent of each
        solvent_counts_list = [solvent_counts[solvent] for solvent in type_counts.index.get_level_values(0)]
        type_percents = type_counts['atom_ix'] / solvent_counts_list
        type_percents.name = 'percent'
        # change index type
        type_percents = (type_percents
                         .reset_index(level=1)
                         .astype({'atom_type': str})
                         .set_index('atom_type', append=True)
                         )
        return type_percents[type_percents.percent > tol]


class Pairing:
    """
    Calculate the percent of solutes that are coordinated with each solvent.

    The pairing percentage is the percent of solutes that are coordinated with
    ANY solvent with matching type. So if the pairing of mol1 is 0.5, then 50% of
    solutes are coordinated with at least 1 mol1.

    The pairing percentages are made available as an mean over the whole
    simulation and by frame.

    Parameters
    ----------
    solvation_data : pandas.DataFrame
        The solvation data frame output by Solution.
    n_frames : int
        The number of frames in solvation_data.
    n_solutes : int
        The number of solutes in solvation_data.
    n_solvents : dict of {str: int}
        The number of each kind of solvent.

    Attributes
    ----------
    pairing_dict : dict of {str: float}
        a dictionary where keys are residue names (str) and values are the
        percentage of solutes that contain that residue (float).
    pairing_by_frame : pd.DataFrame
        a dictionary tracking the mean percentage of each residue across frames.
    percent_free_solvents : dict of {str: float}
        a dictionary containing the percent of each solvent that is free. e.g.
        not coordinated to a solute.
    diluent_dict : dict of {str: float}
        the fraction of the diluent constituted by each solvent. The diluent is
        defined as everything that is not coordinated with the solute.
    diluent_by_frame : pd.DataFrame
        a DataFrame of the diluent composition in each frame of the trajectory.
    diluent_counts : pd.DataFrame
        a DataFrame of the raw solvent counts in the diluent in each frame of the trajectory.


    Examples
    --------

     .. code-block:: python

        # first define Li, BN, and FEC AtomGroups
        >>> solution = Solution(Li, {'BN': BN, 'FEC': FEC, 'PF6': PF6})
        >>> solution.run()
        >>> solution.pairing.pairing_dict
        {'BN': 1.0, 'FEC': 0.210, 'PF6': 0.120}
    """

    def __init__(self, solvation_data, n_frames, n_solutes, n_solvents):
        self.solvation_data = solvation_data
        self.n_frames = n_frames
        self.n_solutes = n_solutes
        self.solvent_counts = n_solvents
        self.pairing_dict, self.pairing_by_frame = self._percent_coordinated()
        self.percent_free_solvents = self._percent_free_solvent()
        self.diluent_dict, self.diluent_by_frame, self.diluent_counts = self._diluent_composition()

    @staticmethod
    def from_solution(solution):
        """
        Generate a Pairing object from a solution.

        Parameters
        ----------
        solution : Solution

        Returns
        -------
        Pairing
        """
        assert solution.has_run, "The solution must be run before calling from_solution"
        return Pairing(
            solution.solvation_data,
            solution.n_frames,
            solution.n_solute,
            solution.solvent_counts
        )

    def _percent_coordinated(self):
        # calculate the percent of solute coordinated with each solvent
        counts = self.solvation_data.groupby(["frame", "solvated_atom", "res_name"]).count()["res_ix"]
        pairing_series = counts.astype(bool).groupby(["res_name", "frame"]).sum() / (
            self.n_solutes
        )  # mean coordinated overall
        pairing_by_frame = pairing_series.unstack()
        pairing_normalized = pairing_series / self.n_frames
        pairing_dict = pairing_normalized.groupby(["res_name"]).sum().to_dict()
        return pairing_dict, pairing_by_frame

    def _percent_free_solvent(self):
        # calculate the percent of each solvent NOT coordinated with the solute
        counts = self.solvation_data.groupby(["frame", "res_ix", "res_name"]).count()['dist']
        totals = counts.groupby(['res_name']).count() / self.n_frames
        n_solvents = np.array([self.solvent_counts[name] for name in totals.index.values])
        free_solvents = np.ones(len(totals)) - totals / n_solvents
        return free_solvents.to_dict()

    def _diluent_composition(self):
        coordinated_solvents = self.solvation_data.groupby(["frame", "res_name"]).nunique()["res_ix"]
        solvent_counts = pd.Series(self.solvent_counts)
        total_solvents = solvent_counts.reindex(coordinated_solvents.index, level=1)
        diluent_solvents = total_solvents - coordinated_solvents
        diluent_series = diluent_solvents / diluent_solvents.groupby(['frame']).sum()
        diluent_by_frame = diluent_series.unstack().T
        diluent_counts = diluent_solvents.unstack().T
        diluent_dict = diluent_by_frame.mean(axis=1).to_dict()
        return diluent_dict, diluent_by_frame, diluent_counts


class Residence:
    """
    Calculate the residence times of solvents.

    This class calculates the residence time of each solvent on the solute.
    The residence time is in units of Solution frames, so if the Solution object
    has 1000 frames over 1 nanosecond, then each frame will be 1 picosecond.
    Thus a residence time of 100 would translate to 100 picoseconds.

    Two residence time implementations are available. Both calculate the
    solute-solvent autocorrelation function for each solute-solvent pair,
    take and take the mean over the solvents of each type, this should yield
    an exponentially decaying autocorrelation function.

    The first implementation fits an exponential curve to the autocorrelation
    function and extract the time constant, which is inversely proportional to the
    residence time. This result is saved in the ``residence_times_fit`` attribute.
    Unfortunately, the fit often fails to converge (value is set to np.nan),
    making this method unreliable.

    Instead, the default implementation is to simply find point where the
    value of the autocorrelation function is 1/e, which is the time constant
    of an exact exponential. These values are saved in ``residence_times``.

    It is recommended that the user visually inspect the autocorrelation function
    with ``Residence.plot_autocorrelation_function`` to ensure an approximately
    exponential decay. The residence times are only valid insofar as the autocorrelation
    function resembles an exact exponential, it should decays to zero with a long tail.
    If the exponential does not decay to zero or its slope does not level off, increasing
    the simulation time may help. For this technique to be appropriate, the simulation time
    should exceed the residence time.

    A fuller description of the method can be found in
    `Self, Fong, and Persson <https://pubs-acs-org.libproxy.berkeley.edu/doi/full/10.1021/acsenergylett.9b02118>`_

    Parameters
    ----------
    solvation_data : pandas.DataFrame
        The solvation data frame output by Solution.
    step : int
        The spacing of frames in solvation_data. This should be equal
        to solution.step.

    Attributes
    ----------
    residence_times : dict of {str: float}
        a dictionary where keys are residue names and values are the
        residence times of the that residue on the solute, calculated
        with the 1/e cutoff method.
    residence_times_fit : dict of {str: float}
        a dictionary where keys are residue names and values are the
        residence times of the that residue on the solute, calculated
        with the exponential fit method.
    fit_parameters : pd.DataFrame
        a dictionary where keys are residue names and values are the
        arameters for the exponential fit to the autocorrelation function.

    Examples
    --------

     .. code-block:: python

        # first define Li, BN, and FEC AtomGroups
        >>> solution = Solution(Li, {'BN': BN, 'FEC': FEC, 'PF6': PF6})
        >>> residence = Residence.from_solution(solution)
        >>> residence.residence_times
        {'BN': 4.02, 'FEC': 3.79, 'PF6': 1.15}
    """

    def __init__(self, solvation_data, step):
        self.solvation_data = solvation_data
        self.auto_covariances = self._calculate_auto_covariance_dict()
        self.residence_times = self._calculate_residence_times_with_cutoff(self.auto_covariances, step)
        self.residence_times_fit, self.fit_parameters = self._calculate_residence_times_with_fit(
            self.auto_covariances,
            step
        )


    @staticmethod
    def from_solution(solution):
        """
        Generate a Residence object from a solution.

        Parameters
        ----------
        solution : Solution

        Returns
        -------
        Residence
        """
        assert solution.has_run, "The solution must be run before calling from_solution"
        return Residence(
            solution.solvation_data,
            solution.step
        )

    def _calculate_auto_covariance_dict(self):
        frame_solute_index = np.unique(self.solvation_data.index.droplevel(2))
        auto_covariance_dict = {}
        for res_name, res_solvation_data in self.solvation_data.groupby(['res_name']):
            adjacency_mini = Residence.calculate_adjacency_dataframe(res_solvation_data)
            adjacency_df = adjacency_mini.reindex(frame_solute_index, fill_value=0)
            auto_covariance = Residence._calculate_auto_covariance(adjacency_df)
            # normalize
            auto_covariance = auto_covariance / np.max(auto_covariance)
            auto_covariance_dict[res_name] = auto_covariance
        return auto_covariance_dict

    @staticmethod
    def _calculate_residence_times_with_cutoff(auto_covariances, step, convergence_cutoff=0.1):
        residence_times = {}
        for res_name, auto_covariance in auto_covariances.items():
            if np.min(auto_covariance) > convergence_cutoff:
                residence_times[res_name] = np.nan
                warnings.warn(f'the autocovariance for {res_name} does not converge to zero '
                              'so a residence time cannot be calculated. A longer simulation '
                              'is required to get a valid estimate of the residence time.')
            unassigned = True
            for frame, val in enumerate(auto_covariance):
                if val < 1 / math.e:
                    residence_times[res_name] = frame * step
                    unassigned = False
                    break
            if unassigned:
                residence_times[res_name] = np.nan
        return residence_times

    @staticmethod
    def _calculate_residence_times_with_fit(auto_covariances, step):
        # calculate the residence times
        residence_times = {}
        fit_parameters = {}
        for res_name, auto_covariance in auto_covariances.items():
            res_time, params = Residence._fit_exponential(auto_covariance, res_name)
            residence_times[res_name], fit_parameters[res_name] = res_time * step, params
        return residence_times, fit_parameters

    def plot_auto_covariance(self, res_name):
        """
        Plot the autocovariance of a solvent on the solute.

        See the discussion in the class docstring for more information.

        Parameters
        ----------
        res_name : str
            the name of a solvent in the solution.

        Returns
        -------
        fig : matplotlib.Figure
        ax : matplotlib.Axes
        """
        auto_covariance = self.auto_covariances[res_name]
        frames = np.arange(len(auto_covariance))
        params = self.fit_parameters[res_name]
        exp_func = lambda x: self._exponential_decay(x, *params)
        exp_fit = np.array(map(exp_func, frames))
        fig, ax = plt.subplots()
        ax.plot(frames, auto_covariance, "b-", label="auto covariance")
        try:
            ax.scatter(frames, exp_fit, label="exponential fit")
        except:
            warnings.warn(f'The fit for {res_name} failed so the exponential '
                          f'fit will not be plotted.')
        ax.hlines(y=1/math.e, xmin=frames[0], xmax=frames[-1], label='1/e cutoff')
        ax.set_xlabel("Timestep (frames)")
        ax.set_ylabel("Normalized Autocovariance")
        ax.set_ylim(0, 1)
        ax.legend()
        return fig, ax

    @staticmethod
    def _exponential_decay(x, a, b, c):
        """
        An exponential decay function.

        Args:
            x: Independent variable.
            a: Initial quantity.
            b: Exponential decay constant.
            c: Constant.

        Returns:
            The acf
        """
        return a * np.exp(-b * x) + c

    @staticmethod
    def _fit_exponential(auto_covariance, res_name):
        auto_covariance_norm = auto_covariance / auto_covariance[0]
        try:
            params, param_covariance = curve_fit(
                Residence._exponential_decay,
                np.arange(len(auto_covariance_norm)),
                auto_covariance_norm,
                p0=(1, 0.1, 0.01),
            )
            tau = 1 / params[1]  # p
        except RuntimeError:
            warnings.warn(f'The fit for {res_name} failed so its values in'
                          f'residence_time_fits and fit_parameters will be'
                          f'set to np.nan.')
            tau, params = np.nan, (np.nan, np.nan, np.nan)
        return tau, params

    @staticmethod
    def _calculate_auto_covariance(adjacency_matrix):
        auto_covariances = []
        for solute_ix, df in adjacency_matrix.groupby(['solvated_atom']):
            non_zero_cols = df.loc[:, (df != 0).any(axis=0)]
            auto_covariance_df = non_zero_cols.apply(
                acovf,
                axis=0,
                result_type='expand',
                demean=False,
                unbiased=True,
                fft=True
            )
            auto_covariances.append(auto_covariance_df.values)
        auto_covariance = np.mean(np.concatenate(auto_covariances, axis=1), axis=1)
        return auto_covariance

    @staticmethod
    def calculate_adjacency_dataframe(solvation_data):
        """
        Calculate a frame-by-frame adjacency matrix from the solvation data.

        This will calculate the adjacency matrix of the solute and all possible
        solvents. It will maintain an index of ['frame', 'solvated_atom', 'res_ix']
        where each 'frame' is a sparse adjacency matrix between solvated atom ix
        and residue ix.

        Parameters
        ----------
        solvation_data : pd.DataFrame
            the solvation_data from a Solution.

        Returns
        -------
        adjacency_df : pandas.DataFrame
        """
        # generate an adjacency matrix from the solvation data
        adjacency_group = solvation_data.groupby(['frame', 'solvated_atom', 'res_ix'])
        adjacency_df = adjacency_group['dist'].count().unstack(fill_value=0)
        return adjacency_df


class Networking:
    """
    Calculate the number and size of solute-solvent networks.

    A network is defined as a bipartite graph of solutes and solvents, where edges
    are defined by coordination in the solvation_data DataFrame. A single solvent
    or multiple solvents can be selected, but coordination between solvents will
    not be included, only coordination between solutes and solvents.

    Networking uses the solvation_data to construct an adjacency matrix and then
    extracts the connected subgraphs within it. These connected subgraphs are stored
    in a DataFrame in Networking.network_df.

    Several other representations of the networking data are included in the attributes.

    Parameters
    ----------
    solvents : str or list[str]
        the solvents to include in the solute-solvent network.
    solvation_data : pandas.DataFrame
        a dataframe of solvation data with columns "frame", "solvated_atom", "atom_ix",
        "dist", "res_name", and "res_ix".
    solute_res_ix : np.ndarray
        the residue indices of the solutes in solvation_data
    res_name_map : pd.Series
        a mapping between residue indices and the solute & solvent names in a Solution.

    Attributes
    ----------
    network_df : pd.DataFrame
        the dataframe containing all networking data. the indices are the frame and
        network index, respectively. the columns are the res_name and res_ix.
    network_sizes : pd.DataFrame
        a dataframe of network sizes. the index is the frame. the column headers
        are network sizes, or the number of solutes + solvents in the network, so
        the columns might be [2, 3, 4, ...]. the values in each column are the
        number of networks with that size in each frame.
    solute_status : dict of {str: float}
        a dictionary where the keys are the "status" of the solute and the values
        are the fraction of solute with that status, averaged over all frames.
        "alone" means that the solute not coordinated with any of the networking
        solvents, network size is 1.
        "paired" means the solute and is coordinated with a single networking
        solvent and that solvent is not coordinated to any other solutes, network
        size is 2.
        "in_network" means that the solute is coordinated to more than one solvent
        or its solvent is coordinated to more than one solute, network size >= 3.
    solute_status_by_frame : pd.DataFrame
        as described above, except organized into a dataframe where each
        row is a unique frame and the columns are "alone", "paired", and "in_network".

    Examples
    --------
     .. code-block:: python

        # first define Li, BN, and FEC AtomGroups
        >>> solution = Solution(Li, {'BN': BN, 'FEC': FEC, 'PF6': PF6})
        >>> networking = Networking.from_solution(solution, 'PF6')
    """

    def __init__(self, solvents, solvation_data, solute_res_ix, res_name_map):
        self.solvents = solvents
        self.solvation_data = solvation_data
        solvent_present = np.isin(self.solvents, self.solvation_data['res_name'].unique())
        if not solvent_present.all():
            raise Exception(f"Solvent(s) {np.array(self.solvents)[~solvent_present]} not found in solvation data.")
        self.solute_res_ix = solute_res_ix
        self.res_name_map = res_name_map
        self.n_solute = len(solute_res_ix)
        self.network_df = self._generate_networks()
        self.network_sizes = self._calculate_network_sizes()
        self.solute_status, self.solute_status_by_frame = self._calculate_solute_status()
        self.solute_status = self.solute_status.to_dict()

    @staticmethod
    def from_solution(solution, solvents):
        """
        Generate a Networking object from a solution and solvent names.

        Parameters
        ----------
        solution : Solution
        solvents : str or list of str
            the strings should be the name of solvents in the Solution. The
            strings must match exactly for Networking to work properly. The
            selected solvents will be used to construct the networking graph
            that is described in documentation for the Networking class.

        Returns
        -------
        Networking
        """
        return Networking(
            solvents,
            solution.solvation_data,
            solution.solute_res_ix,
            solution.res_name_map,
        )

    @staticmethod
    def _unwrap_adjacency_dataframe(df):
        # this class will transform the biadjacency matrix into a proper adjacency matrix
        connections = df.reset_index(level=0).drop(columns='frame')
        idx = connections.columns.append(connections.index)
        directed = connections.reindex(index=idx, columns=idx, fill_value=0)
        undirected = directed.values + directed.values.T
        adjacency_matrix = csr_matrix(undirected)
        return adjacency_matrix

    def _generate_networks(self):
        """
        This function generates a dataframe containing all the solute-solvent networks
        in every frame of the simulation. The rough approach is as follows:

        1. transform the solvation_data DataFrame into an adjacency matrix
        2. determine the connected subgraphs in the adjacency matrix
        3. tabulate the res_ix involved in each network and store in a DataFrame
        """
        solvents = [self.solvents] if isinstance(self.solvents, str) else self.solvents
        solvation_subset = self.solvation_data[np.isin(self.solvation_data.res_name, solvents)]
        # reindex solvated_atom to residue indexes
        reindexed_subset = solvation_subset.reset_index(level=1)
        reindexed_subset.solvated_atom = self.solute_res_ix[reindexed_subset.solvated_atom]
        dropped_reindexed = reindexed_subset.set_index(['solvated_atom'], append=True)
        reindexed_subset = dropped_reindexed.reorder_levels(['frame', 'solvated_atom', 'atom_ix'])
        # create adjacency matrix from reindexed df
        graph = Residence.calculate_adjacency_dataframe(reindexed_subset)
        network_arrays = []
        # loop through each time step / frame
        for frame, df in graph.groupby('frame'):
            # drop empty columns
            df = df.loc[:, (df != 0).any(axis=0)]
            # save map from local index to residue index
            solute_map = df.index.get_level_values(1).values
            solvent_map = df.columns.values
            ix_to_res_ix = np.concatenate([solvent_map, solute_map])
            adjacency_df = Networking._unwrap_adjacency_dataframe(df)
            _, network = connected_components(
                csgraph=adjacency_df,
                directed=False,
                return_labels=True
            )
            network_array = np.vstack([
                np.full(len(network), frame),  # frame
                network,  # network
                self.res_name_map[ix_to_res_ix],  # res_names
                ix_to_res_ix,  # res index
            ]).T
            network_arrays.append(network_array)
        # create and return network dataframe
        all_clusters = np.concatenate(network_arrays)
        cluster_df = (
            pd.DataFrame(all_clusters, columns=['frame', 'network', 'res_name', 'res_ix'])
                .set_index(['frame', 'network'])
                .sort_values(['frame', 'network'])
        )
        return cluster_df

    def _calculate_network_sizes(self):
        # This utility calculates the network sizes and returns a convenient dataframe.
        cluster_df = self.network_df
        cluster_sizes = cluster_df.groupby(['frame', 'network']).count()
        size_counts = cluster_sizes.groupby(['frame', 'res_name']).count().unstack(fill_value=0)
        size_counts.columns = size_counts.columns.droplevel()
        return size_counts

    def _calculate_solute_status(self):
        """
        This utility calculates the percentage of each solute with a given "status".
        Namely, whether the solvent is "alone", "paired" (with a single solvent), or
        "in_network" of > 2 species.
        """
        status = self.network_sizes.rename(columns={2: 'paired'})
        status['in_network'] = status.iloc[:, 1:].sum(axis=1).astype(int)
        status['alone'] = self.n_solute - status.loc[:, ['paired', 'in_network']].sum(axis=1)
        status = status.loc[:, ['alone', 'paired', 'in_network']]
        solute_status_by_frame = status / self.n_solute
        solute_status = solute_status_by_frame.mean()
        return solute_status, solute_status_by_frame

    def get_network_res_ix(self, network_index, frame):
        """
        Return the indexes of all residues in a selected network.

        The network_index and frame must be provided to fully specify the network.
        Once the indexes are returned, they can be used to select an AtomGroup with
        the species of interest, see Examples.

        Parameters
        ----------
        network_index : int
            The index of the network of interest
        frame : int
            the frame in the trajectory to perform selection at. Defaults to the
            current trajectory frame.
        Returns
        -------
        res_ix : np.ndarray

        Examples
        --------
         .. code-block:: python

            # first define Li, BN, and FEC AtomGroups
            >>> solution = Solution(Li, {'BN': BN, 'FEC': FEC, 'PF6': PF6})
            >>> networking = Networking.from_solution(solution, 'PF6')
            >>> res_ix = networking.get_network_res_ix(1, 5)
            >>> solution.u.residues[res_ix].atoms
            <AtomGroup with 126 Atoms>

        """
        res_ix = self.network_df.loc[pd.IndexSlice[frame, network_index], 'res_ix'].values
        return res_ix.astype(int)
