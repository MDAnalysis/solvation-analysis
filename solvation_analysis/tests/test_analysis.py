import matplotlib.pyplot as plt
import pytest
import copy

from copy import deepcopy

from solvation_analysis.analysis import Solution
import numpy as np


def test_plot_solvation_distance(rdf_bins_and_data_easy):
    bins, data = rdf_bins_and_data_easy['pf6_all']
    fig, ax = Solution._plot_solvation_radius(bins, data, 2)
    # plt.show()  # comment out for global testing


def test_instantiate_solute(pre_solution):
    # these check basic properties of the instantiation
    assert len(pre_solution.radii) == 1
    assert len(pre_solution.rdf_data) == 0
    assert len(pre_solution.rdf_plots) == 0
    assert callable(pre_solution.kernel)
    assert pre_solution.solute.n_residues == 49
    assert pre_solution.solvents['pf6'].n_residues == 49
    assert pre_solution.solvents['fec'].n_residues == 237
    assert pre_solution.solvents['bn'].n_residues == 363


def test_radii_finding(run_solution):
    # checks that the solvation radii are plotted
    assert len(run_solution.radii) == 3
    assert len(run_solution.rdf_data) == 3
    assert len(run_solution.rdf_plots) == 3
    # checks that the identified solvation radii are approximately correct
    assert 2 < run_solution.radii['pf6'] < 3
    assert 2 < run_solution.radii['fec'] < 3
    assert 2 < run_solution.radii['bn'] < 3
    # for fig, ax in run_solute.rdf_plots.values():
    #     plt.show()  # comment out for global testing


def test_run(run_solution):
    # checks that run is run with
    assert len(run_solution.solvation_frames) == 10
    assert len(run_solution.solvation_frames[0]) == 228


@pytest.mark.parametrize(
    "step_size, index_for_2, index_for_9",
    [
        (1, 2, 9),
        (2, 1, 4),
        (3, 0, 3),
    ],
)
def test_map_step_to_index(step_size, index_for_2, index_for_9, pre_solution_mutable):
    pre_solution_mutable.run(step=step_size)
    assert pre_solution_mutable.map_step_to_index(2) == index_for_2
    assert pre_solution_mutable.map_step_to_index(9) == index_for_9

# @pytest.mark.parametrize(
#     "solute_index, n_mol, step",
#     [
#         (1, 5, 3),
#         (2, 6, [3773, 3173, 161, 713, 2129]),
#         (40, 0, [4325, 1517, 1217, 1529])
#     ],
# )
# def test_radial_shell(run_solution):
#     return
#
# @pytest.mark.parametrize(
#     "solute_index, radius, step",
#     [
#         (1, 5, [3065, 557, 2057, 1205]),
#         (2, 6, [3773, 3173, 161, 713, 2129]),
#         (40, 0, [4325, 1517, 1217, 1529])
#     ],
# )
# def test_closest_n_mol(run_solution):
#     return


@pytest.mark.parametrize(
    "solute_index, step, expected_res_ids",
    [
        (1, 5, [47, 101, 172, 256]),
        (2, 6, [14, 60, 178, 265, 315]),
        (40, 0, [102, 127, 128, 361])
    ],
)
def test_solvation_shell(solute_index, step, expected_res_ids, run_solution):
    shell = run_solution.solvation_shell(solute_index, step)
    assert set(shell.resids) == set(expected_res_ids)


def test_selection_functions(run_solution):
    # this test is incomplete and is currently only demonstrating functionality

    run_solution.radial_shell(31, 3)       # 31 is a local solute index, 3 is a radius
    run_solution.closest_n_mol(31, 6)      # 6 is n_mol
    run_solution.solvation_shell(31, 9)  # 510 is the trajectory step of interest

# TODO: should test what happens when the solute is included as a solvent
