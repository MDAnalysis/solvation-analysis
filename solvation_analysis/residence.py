"""
================
Residence
================
:Author: Orion Cohen, Tingzheng Hou, Kara Fong
:Year: 2021
:Copyright: GNU Public License v3

Understand the dynamic coordination of solvents with the solute.

Residence times for all solvents are automatically calculated from autocovariance
of the solvent-solute adjacency matrix.

While ``residence`` can be used in isolation, it is meant to be used
as an attribute of the Solute class. This makes instantiating it and calculating the
solvation data a non-issue.
"""
import math
import warnings

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import acovf
from scipy.optimize import curve_fit

from solvation_analysis._column_names import *
from solvation_analysis._utils import calculate_adjacency_dataframe


class Residence:
    """
    Calculate the residence times of solvents.

    This class calculates the residence time of each solvent on the solute.
    The residence time is in units of Solute frames, so if the Solute object
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
        The solvation data frame output by Solute.
    step : int
        The spacing of frames in solvation_data. This should be equal
        to solute.step.

    Examples
    --------

     .. code-block:: python

        # first define Li, BN, and FEC AtomGroups
        >>> solute = Solute(Li, {'BN': BN, 'FEC': FEC, 'PF6': PF6})
        >>> residence = Residence.from_solute(solute)
        >>> residence.residence_times_cutoff
        {'BN': 4.02, 'FEC': 3.79, 'PF6': 1.15}
    """

    def __init__(self, solvation_data, step):
        self.solvation_data = solvation_data
        self._auto_covariances = self._calculate_auto_covariance_dict()
        self._residence_times_cutoff = self._calculate_residence_times_with_cutoff(self._auto_covariances, step)
        self._residence_times_fit, self._fit_parameters = self._calculate_residence_times_with_fit(
            self._auto_covariances,
            step
        )

    @staticmethod
    def from_solute(solute):
        """
        Generate a Residence object from a solute.

        Parameters
        ----------
        solute : Solute

        Returns
        -------
        Residence
        """
        assert solute.has_run, "The solute must be run before calling from_solute"
        return Residence(
            solute.solvation_data,
            solute.step
        )

    def _calculate_auto_covariance_dict(self):
        partial_index = self.solvation_data.index.droplevel(SOLVENT_ATOM_IX)
        unique_indices = np.unique(partial_index)
        frame_solute_index = pd.MultiIndex.from_tuples(unique_indices, names=partial_index.names)
        auto_covariance_dict = {}
        for res_name, res_solvation_data in self.solvation_data.groupby([SOLVENT]):
            if isinstance(res_name, tuple):
                res_name = res_name[0]
            adjacency_mini = calculate_adjacency_dataframe(res_solvation_data)
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
            residence_times[res_name], fit_parameters[res_name] = round(res_time * step, 2), params
        return residence_times, fit_parameters

    def plot_auto_covariance(self, res_name):
        """
        Plot the autocovariance of a solvent on the solute.

        See the discussion in the class docstring for more information.

        Parameters
        ----------
        res_name : str
            the name of a solvent in the solute.

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
        timesteps = adjacency_matrix.index.levels[0]

        for solute_ix, solute_df in adjacency_matrix.groupby([SOLUTE_IX, SOLUTE_ATOM_IX]):
            # this is needed to make sure auto-covariances can be concatenated later
            new_solute_df = solute_df.droplevel([SOLUTE_IX, SOLUTE_ATOM_IX]).reindex(timesteps, fill_value=0)
            non_zero_cols = new_solute_df.loc[:, (solute_df != 0).any(axis=0)]
            auto_covariance_df = non_zero_cols.apply(
                acovf,
                axis=0,
                result_type='expand',
                demean=False,
                adjusted=True,
                fft=True
            )
            # timesteps with no binding are getting skipped, we need to make sure to include all timesteps
            auto_covariances.append(auto_covariance_df.values)

        auto_covariance = np.mean(np.concatenate(auto_covariances, axis=1), axis=1)
        return auto_covariance

    @property
    def auto_covariances(self):
        """
        A dictionary where keys are residue names and values are the
        autocovariance of the that residue on the solute.
        """
        return self._auto_covariances

    @property
    def residence_times_cutoff(self):
        """
        A dictionary where keys are residue names and values are the
        residence times of the that residue on the solute, calculated
        with the 1/e cutoff method.
        """
        return self._residence_times_cutoff

    @property
    def residence_times_fit(self):
        """
        A dictionary where keys are residue names and values are the
        residence times of the that residue on the solute, calculated
        with the exponential fit method.
        """
        return self._residence_times_fit

    @property
    def fit_parameters(self):
        """
        A dictionary where keys are residue names and values are the
        arameters for the exponential fit to the autocorrelation function.
        """
        return self._fit_parameters
