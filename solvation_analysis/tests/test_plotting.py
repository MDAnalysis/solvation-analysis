import numpy as np
import pytest

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

def test_compare_pairing_1(eax_solutions):
    fig = compare_pairing(eax_solutions, "species", False, ["fec", "pf6"])
    fig = format_graph(fig, "Line Graph of Solvent Pairing", "Species", "Pairing")
    fig.show()
    assert True

def test_compare_pairing_2(eax_solutions):
    fig = compare_pairing(eax_solutions,"solution", False, ["pf6", "fec"])
    fig = format_graph(fig, "Line Graph of Solvent Pairing", "Solution", "Pairing")
    fig.show()
    assert True

def test_compare_coordination_numbers(eax_solutions):
    fig = compare_coordination_numbers(eax_solutions)
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

