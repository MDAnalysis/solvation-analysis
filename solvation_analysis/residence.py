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
as an attribute of the Solution class. This makes instantiating it and calculating the
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
        for solute_ix, df in adjacency_matrix.groupby([SOLVATED_ATOM]):
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
        solvents. It will maintain an index of ["frame", 'solvated_atom', 'res_ix']
        where each "frame" is a sparse adjacency matrix between solvated atom ix
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
        adjacency_group = solvation_data.groupby([FRAME, SOLVATED_ATOM, 'res_ix'])
        adjacency_df = adjacency_group[DISTANCE].count().unstack(fill_value=0)
        return adjacency_df

