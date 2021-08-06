import matplotlib.pyplot as plt
import pytest

from copy import deepcopy

from solvation_analysis.analysis import Solution
import numpy as np


def test_plot_solvation_distance(rdf_bins_and_data_easy):
    bins, data = rdf_bins_and_data_easy['pf6_all']
    fig, ax = Solution._plot_solvation_radius(bins, data, 2)
    # plt.show()  # comment out for global testing


def test_instantiate_solute(default_solute):
    # these check basic properties of the instantiation
    assert len(default_solute.radii) == 1
    assert len(default_solute.rdf_data) == 0
    assert len(default_solute.rdf_plots) == 0
    assert callable(default_solute.kernel)
    assert default_solute.solute.n_residues == 49
    assert default_solute.solvents['pf6'].n_residues == 49
    assert default_solute.solvents['fec'].n_residues == 237
    assert default_solute.solvents['bn'].n_residues == 363


def test_radii_finding(run_solute):
    # checks that the solvation radii are plotted
    assert len(run_solute.radii) == 3
    assert len(run_solute.rdf_data) == 3
    assert len(run_solute.rdf_plots) == 3
    # checks that the identified solvation radii are approximately correct
    assert 2 < run_solute.radii['pf6'] < 3
    assert 2 < run_solute.radii['fec'] < 3
    assert 2 < run_solute.radii['bn'] < 3
    # for fig, ax in run_solute.rdf_plots.values():
    #     plt.show()  # comment out for global testing


def test_run(run_solute):
    # checks that run is run with
    assert len(run_solute.solvation_frames) == 10
    assert len(run_solute.solvation_frames[0]) == 228


def test_selection_functions(run_solute):
    # this test is incomplete and is currently only demonstrating functionality

    run_solute.radial_shell(31, 3)       # 31 is a local solute index, 3 is a radius
    run_solute.closest_n_mol(31, 6)      # 6 is n_mol
    run_solute.solvation_shell(31, 9)  # 510 is the trajectory step of interest

# TODO: should test what happens when the solute is included as a solvent
