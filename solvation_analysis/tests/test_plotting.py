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


# TODO: solution + species names are case sensitive; what happens if user specifies wrongly formatted strings?
def test_compare_pairing_default_eax(eax_solutions):
    # call compare_pairing with only one required argument
    fig = compare_pairing(eax_solutions)
    assert len(fig.data) == 4
    fig.show()


def test_compare_pairing_default(run_solution):
    fig = compare_pairing(run_solution)
    fig.show()


def test_compare_pairing_1(eax_solutions):
    # keep_solvents on x axis, each bar is a solution
    fig = compare_pairing(eax_solutions, keep_solvents=["fec", "pf6"], x_label="Species", y_label="Pairing", title="Bar Graph of Solvent Pairing",)
    assert len(fig.data) == 4
    for bar in fig.data:
        assert set(bar.x) == {"fec", "pf6"}
    fig.show()


def test_compare_pairing_2(eax_solutions):
    # solutions on x axis, each bar is an element of keep_solvents
    fig = compare_pairing(eax_solutions, keep_solvents=["pf6", "fec"], x_label="Solution", y_label="Pairing", title="Bar Graph of Solvent Pairing", x_axis="solution")
    assert len(fig.data) == 2
    for bar in fig.data:
        assert set(bar.x) == {"feaf", "eaf", "fea", "ea"}
    fig.show()


def test_compare_pairing_3(eax_solutions):
    # keep_solvents on x axis, each line is a solution
    fig = compare_pairing(eax_solutions, keep_solvents=["pf6", "fec"], x_label="Solution", y_label="Pairing", title="Line Graph of Solvent Pairing",series=True)
    assert len(fig.data) == 4
    for line in fig.data:
        assert set(line.x) == {"fec", "pf6"}
    fig.show()


def test_compare_pairing_4(eax_solutions):
    # solutions on x axis, each line is an element of keep_solvents
    fig = compare_pairing(eax_solutions, keep_solvents=["pf6", "fec"], x_label="Solution", y_label="Pairing", title="Line Graph of Solvent Pairing", x_axis="solution", series=True)
    assert len(fig.data) == 2
    for line in fig.data:
        assert set(line.x) == {"feaf", "eaf", "fea", "ea"}
    fig.show()


def test_compare_pairing_5(eax_solutions):
    # same test as test_compare_pairing_4, except order for keep_solvents is switched
    fig = compare_pairing(eax_solutions, keep_solvents=["fec", "pf6"], x_label="Solution", y_label="Pairing", title="Line Graph of Solvent Pairing", x_axis="solution", series=True)
    assert len(fig.data) == 2
    for line in fig.data:
        assert set(line.x) == {"feaf", "eaf", "fea", "ea"}
    fig.show()


def test_compare_pairing_coerce(eax_solutions):
    fig = compare_pairing(eax_solutions, coerce={"ea": "EAx", "fea": "EAx", "eaf": "EAx", "feaf": "EAx"},
                          keep_solvents=["pf6", "fec", "EAx"], x_label="Species", y_label="Pairing", title="Bar Graph of Solvent Pairing")
    assert len(fig.data) == 4
    for bar in fig.data:
        assert set(bar.x) == {"pf6", "fec", "EAx"}
    fig.show()


def test_compare_pairing_6(eax_solutions):
    with pytest.raises(Exception):
        fig = compare_pairing(eax_solutions, coerce={"ea": "EAx", "fea": "EAx", "eaf": "EAx", "feaf": "EAx"},
                          keep_solvents=["pf6", "fec", "ea", "fea", "eaf", "feaf"], x_label="Species", y_label="Pairing", title="Graph")


def test_compare_coordination_numbers(eax_solutions):
    fig = compare_coordination_numbers(eax_solutions, "Species", "Pairing", "Bar Graph of Coordination Numbers", keep_solvents=["fec", "pf6"])
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

