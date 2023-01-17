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
    compare_solvent_dicts,
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


# compare_solvent_dicts tests
def test_compare_solvent_dicts_coerce_exception(eax_solutions):
    # invalid keep_solvents because solvent names were already coerced to the generic "EAx" form
    # keep_solvents here references the former names of solvents, which is wrong
    # this test should handle an exception
    with pytest.raises(Exception):
        fig = compare_pairing(eax_solutions, coerce_solvent_names={"ea": "EAx", "fea": "EAx", "eaf": "EAx", "feaf": "EAx"},
                          keep_solvents=["pf6", "fec", "ea", "fea", "eaf", "feaf"], x_label="Species", y_label="Pairing", title="Graph")


def test_compare_solvent_dicts_sensitivity(eax_solutions):
    # solvent names are case-sensitive, so names in keep_solvents and coerce_solvent_names should be consistent
    # this test should handle an exception
    with pytest.raises(Exception):
        fig = compare_pairing(eax_solutions, coerce_solvent_names={"EA": "EAx", "fEA": "EAx", "EAf": "EAx", "fEAf": "EAx"},
                          keep_solvents=["PF6", "FEC", "EAx"], x_label="Species", y_label="Pairing", title="Graph")


# compare_pairing tests
# TODO: complete this test
def test_compare_pairing_default(run_solution):
    # call compare_pairing with only one required argument
    fig = compare_pairing(run_solution)
    fig.show()


def test_compare_pairing_default_eax(eax_solutions):
    # call compare_pairing with only one required argument
    # also tests how the code handles eax systems
    fig = compare_pairing(eax_solutions)
    assert len(fig.data) == 4
    fig.show()


def test_compare_pairing_case1(eax_solutions):
    # keep_solvents on x axis, each bar is a solution
    fig = compare_pairing(eax_solutions, keep_solvents=["fec", "pf6"], x_label="Species", y_label="Pairing", title="Bar Graph of Solvent Pairing")
    assert len(fig.data) == 4
    for bar in fig.data:
        assert set(bar.x) == {"fec", "pf6"}
    fig.show()


def test_compare_pairing_case2(eax_solutions):
    # solutions on x axis, each bar is an element of keep_solvents
    fig = compare_pairing(eax_solutions, keep_solvents=["pf6", "fec"], x_label="Solution", y_label="Pairing", title="Bar Graph of Solvent Pairing", x_axis="solution")
    assert len(fig.data) == 2
    for bar in fig.data:
        assert set(bar.x) == {"feaf", "eaf", "fea", "ea"}
    fig.show()


def test_compare_pairing_case3(eax_solutions):
    # keep_solvents on x axis, each line is a solution
    fig = compare_pairing(eax_solutions, keep_solvents=["pf6", "fec"], x_label="Solution", y_label="Pairing", title="Line Graph of Solvent Pairing",series=True)
    assert len(fig.data) == 4
    for line in fig.data:
        assert set(line.x) == {"fec", "pf6"}
    fig.show()


def test_compare_pairing_case4(eax_solutions):
    # solutions on x axis, each line is an element of keep_solvents
    fig = compare_pairing(eax_solutions, keep_solvents=["pf6", "fec"], x_label="Solution", y_label="Pairing", title="Line Graph of Solvent Pairing", x_axis="solution", series=True)
    assert len(fig.data) == 2
    for line in fig.data:
        assert set(line.x) == {"feaf", "eaf", "fea", "ea"}
    fig.show()


def test_compare_pairing_switch_keep_solvents_order(eax_solutions):
    # same test as test_compare_pairing_case4, except order for keep_solvents is switched
    fig = compare_pairing(eax_solutions, keep_solvents=["fec", "pf6"], x_label="Solution", y_label="Pairing", title="Line Graph of Solvent Pairing", x_axis="solution", series=True)
    assert len(fig.data) == 2
    for line in fig.data:
        assert set(line.x) == {"feaf", "eaf", "fea", "ea"}
    fig.show()


def test_compare_pairing_coerce(eax_solutions):
    # coerce solvent names into the generic "EAx" form
    fig = compare_pairing(eax_solutions, coerce_solvent_names={"ea": "EAx", "fea": "EAx", "eaf": "EAx", "feaf": "EAx"},
                          keep_solvents=["pf6", "fec", "EAx"], x_label="Species", y_label="Pairing", title="Bar Graph of Solvent Pairing")
    assert len(fig.data) == 4
    for bar in fig.data:
        assert set(bar.x) == {"pf6", "fec", "EAx"}
    fig.show()


# compare_coordination_numbers tests
# TODO: complete this test
def test_compare_coordination_numbers_default(run_solution):
    # call compare_coordination_numbers with only one required argument
    assert True


def test_compare_coordination_numbers_default_eax(eax_solutions):
    # call compare_coordination_numbers with only one required argument
    # also tests how the code handles eax systems
    fig = compare_coordination_numbers(eax_solutions)
    assert len(fig.data) == 4
    fig.show()


def test_compare_coordination_numbers_case1(eax_solutions):
    # keep_solvents on x axis, each bar is a solution
    fig = compare_coordination_numbers(eax_solutions, keep_solvents=["fec", "pf6"], x_label="Species", y_label="Coordination",
                          title="Bar Graph of Coordination Numbers")
    assert len(fig.data) == 4
    for bar in fig.data:
        assert set(bar.x) == {"fec", "pf6"}
    fig.show()

def test_compare_coordination_numbers_case2(eax_solutions):
    # solutions on x axis, each bar is an element of keep_solvents
    fig = compare_coordination_numbers(eax_solutions, keep_solvents=["pf6", "fec"], x_label="Solution", y_label="Coordination",
                          title="Bar Graph of Coordination Numbers", x_axis="solution")
    assert len(fig.data) == 2
    for bar in fig.data:
        assert set(bar.x) == {"feaf", "eaf", "fea", "ea"}
    fig.show()


def test_compare_coordination_numbers_case3(eax_solutions):
    # keep_solvents on x axis, each line is a solution
    fig = compare_coordination_numbers(eax_solutions, keep_solvents=["pf6", "fec"], x_label="Solution", y_label="Coordination",
                          title="Line Graph of Coordination Numbers", series=True)
    assert len(fig.data) == 4
    for line in fig.data:
        assert set(line.x) == {"fec", "pf6"}
    fig.show()


def test_compare_coordination_numbers_case4(eax_solutions):
    # solutions on x axis, each line is an element of keep_solvents
    fig = compare_coordination_numbers(eax_solutions, keep_solvents=["pf6", "fec"], x_label="Solution", y_label="Coordination",
                          title="Line Graph of Coordination Numbers", x_axis="solution", series=True)
    assert len(fig.data) == 2
    for line in fig.data:
        assert set(line.x) == {"feaf", "eaf", "fea", "ea"}
    fig.show()


# compare_residence_times tests
# TODO: complete this test
def test_compare_residence_times_default(run_solution):
    assert True


def test_compare_residence_times_default_eax(eax_solutions):
    # call compare_residence_times with only one required argument
    # also tests how the code handles eax systems
    fig = compare_pairing(eax_solutions)
    assert len(fig.data) == 4
    fig.show()


def test_compare_residence_times_case1(eax_solutions):
    # keep_solvents on x axis, each bar is a solution
    fig = compare_residence_times(eax_solutions, keep_solvents=["fec", "pf6"], x_label="Species", y_label="Residence Times",
                          title="Bar Graph of Residence Times")
    assert len(fig.data) == 4
    for bar in fig.data:
        assert set(bar.x) == {"fec", "pf6"}
    fig.show()


def test_compare_residence_times_case2(eax_solutions):
    # solutions on x axis, each bar is an element of keep_solvents
    fig = compare_residence_times(eax_solutions, keep_solvents=["pf6", "fec"], x_label="Solution",
                                       y_label="Residence Times",
                                       title="Bar Graph of Residence Times", x_axis="solution")
    assert len(fig.data) == 2
    for bar in fig.data:
        assert set(bar.x) == {"feaf", "eaf", "fea", "ea"}
    fig.show()


def test_compare_residence_times_case3(eax_solutions):
    # keep_solvents on x axis, each line is a solution
    fig = compare_residence_times(eax_solutions, keep_solvents=["pf6", "fec"], x_label="Solution",
                                       y_label="Residence Times",
                                       title="Line Graph of Residence Times", series=True)
    assert len(fig.data) == 4
    for line in fig.data:
        assert set(line.x) == {"fec", "pf6"}
    fig.show()


def test_compare_residence_times_case4(eax_solutions):
    # solutions on x axis, each line is an element of keep_solvents
    fig = compare_residence_times(eax_solutions, keep_solvents=["pf6", "fec"], x_label="Solution",
                                       y_label="Residence Times",
                                       title="Line Graph of Residence Times", x_axis="solution", series=True)
    assert len(fig.data) == 2
    for line in fig.data:
        assert set(line.x) == {"feaf", "eaf", "fea", "ea"}
    fig.show()


def test_compare_residence_times_res_type(eax_solutions):
    # keep_solvents on x axis, each bar is a solution
    fig = compare_residence_times(eax_solutions, res_type="residence_time", keep_solvents=["fec", "pf6"], x_label="Species",
                                  y_label="Residence Times",
                                  title="Bar Graph of Residence Times")
    assert len(fig.data) == 4
    for bar in fig.data:
        assert set(bar.x) == {"fec", "pf6"}
    fig.show()


def test_compare_residence_times_exception(eax_solutions):
    # this test should handle an exception relating to the acceptable arguments for res_type
    with pytest.raises(Exception):
        fig = compare_residence_times(eax_solutions, res_type="residence time", keep_solvents=["fec", "pf6"],
                                      x_label="Species",
                                      y_label="Residence Times",
                                      title="Bar Graph of Residence Times")


# compare_speciation tests
# TODO: complete this test
def test_compare_speciation_default(eax_solutions):
    fig = compare_speciation(eax_solutions)
    fig.show()
    assert True


def test_compare_speciation_default_eax(eax_solutions):
    # call compare_speciation with only one required argument
    # also tests how the code handles eax systems
    fig = compare_speciation(eax_solutions)
    assert len(fig.data) == 4
    fig.show()


def test_compare_speciation_case1(eax_solutions):
    # keep_solvents on x axis, each bar is a solution
    fig = compare_speciation(eax_solutions, keep_solvents=["fec", "pf6"], x_label="Species",
                                       y_label="Speciation",
                                       title="Bar Graph of Speciation")
    assert len(fig.data) == 4
    for bar in fig.data:
        assert set(bar.x) == {"fec", "pf6"}
    fig.show()


def test_compare_speciation_case2(eax_solutions):
    # solutions on x axis, each bar is an element of keep_solvents
    fig = compare_speciation(eax_solutions, keep_solvents=["pf6", "fec"], x_label="Solution",
                                       y_label="Speciation",
                                       title="Bar Graph of Speciation", x_axis="solution")
    assert len(fig.data) == 2
    for bar in fig.data:
        assert set(bar.x) == {"feaf", "eaf", "fea", "ea"}
    fig.show()


def test_compare_speciation_case3(eax_solutions):
    # keep_solvents on x axis, each line is a solution
    fig = compare_speciation(eax_solutions, keep_solvents=["pf6", "fec"], x_label="Solution",
                                       y_label="Speciation",
                                       title="Line Graph of Speciation", series=True)
    assert len(fig.data) == 4
    for line in fig.data:
        assert set(line.x) == {"fec", "pf6"}
    fig.show()


def test_compare_speciation_case4(eax_solutions):
    # solutions on x axis, each line is an element of keep_solvents
    fig = compare_coordination_numbers(eax_solutions, keep_solvents=["pf6", "fec"], x_label="Solution",
                                       y_label="Speciation",
                                       title="Line Graph of Speciation", x_axis="solution", series=True)
    assert len(fig.data) == 2
    for line in fig.data:
        assert set(line.x) == {"feaf", "eaf", "fea", "ea"}
    fig.show()

