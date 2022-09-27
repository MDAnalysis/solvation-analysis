import numpy as np
import pytest
from solvation_analysis.plotting import *

from solvation_analysis.plotting import (
    plot_network_size_histogram,
    plot_shell_size_histogram,
    compare_pairing,
    compare_coordination_numbers,
    compare_residence_times,
    compare_speciation,
    format_graph,
)


def test_plot_network_size_histogram(networking):
    fig = plot_network_size_histogram(networking)
    fig.show()
    assert True

def test_plot_shell_size_histogram(run_solution):
    fig = plot_shell_size_histogram(run_solution)
    fig.show()
    assert True

# just skeleton code, fill in later. make multiple tests for catch_different_solvents
# to make sure it works in all cases as intended
def test_catch_different_solvents(insert_solvent_here):
    catch_different_solvents(insert_solvent_here)

def test_compare_pairing_1(eax_solutions):
    # keep_solvents on x axis, each bar is a solution
    fig = compare_pairing(eax_solutions, ["fec", "pf6"])
    fig = format_graph(fig, "Bar Graph of Solvent Pairing", "Species", "Pairing")
    fig.show()
    assert True

def test_compare_pairing_2(eax_solutions):
    # solutions on x axis, each bar is an element of keep_solvents
    fig = compare_pairing(eax_solutions, ["pf6", "fec"], x_axis="solution")
    fig = format_graph(fig, "Bar Graph of Solvent Pairing", "Solution", "Pairing")
    fig.show()
    assert True

def test_compare_pairing_3(eax_solutions):
    # keep_solvents on x axis, each line is a solution
    fig = compare_pairing(eax_solutions,["pf6", "fec"], series=True)
    fig = format_graph(fig, "Line Graph of Solvent Pairing", "Solution", "Pairing")
    fig.show()
    assert True

def test_compare_pairing_4(eax_solutions):
    # solutions on x axis, each line is an element of keep_solvents
    fig = compare_pairing(eax_solutions, ["pf6", "fec"], x_axis="solution", series=True)
    fig = format_graph(fig, "Line Graph of Solvent Pairing", "Solution", "Pairing")
    fig.show()
    assert True

# it would be nice if we had a list of solutions, all of the same composition,
# and tested compare_pairing to see if the default behavior of keep_solvents works as expected
# def test_compare_pairing_5(solutions):
#     fig = compare_pairing(solutions)
#     fig.show()
#     assert True

def test_compare_coordination_numbers(eax_solutions):
    fig = compare_coordination_numbers(eax_solutions, ["fec", "pf6"])
    fig = format_graph(fig, "Bar Graph of Coordination Numbers", "Species", "Coordination")
    fig.show()
    assert True

def test_compare_residence_times(eax_solutions):
    fig = compare_residence_times(eax_solutions)
    fig.show()
    assert True

def test_compare_speciation(eax_solutions):
    fig = compare_speciation(eax_solutions)
    fig.show()
    assert True

