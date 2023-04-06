import pytest
from solvation_analysis.plotting import (
    plot_network_size_histogram,
    plot_shell_composition_by_size,
    plot_co_occurrence,
    _compare_function_generator,
    compare_free_solvents,
    compare_pairing,
    compare_coordination_numbers,
    compare_residence_times_cutoff,
    compare_residence_times_fit,
    compare_diluent,
)

from solvation_analysis.networking import Networking
from solvation_analysis.residence import Residence
from solvation_analysis.speciation import Speciation


def test_plot_network_size_histogram(run_solute):
    run_solute.networking = Networking.from_solute(run_solute, 'pf6')
    plot_network_size_histogram(run_solute)
    plot_network_size_histogram(run_solute.networking)
    assert True


def test_plot_shell_size_histogram(run_solute):
    plot_shell_composition_by_size(run_solute)
    plot_shell_composition_by_size(run_solute.speciation)
    assert True


# compare_solvent_dicts tests
def test_compare_solvent_dicts_rename_exception(eax_solutes):
    # invalid solvents_to_plot because solvent names were already renamed to the generic "EAx" form
    # solvents_to_plot here references the former names of solvents, which is wrong
    # this test should handle an exception
    with pytest.raises(Exception):
        fig = compare_pairing(
            eax_solutes,
            rename_solvent_dict={
                "ea": "EAx",
                "fea": "EAx",
                "eaf": "EAx",
                "feaf": "EAx",
            },
            solvents_to_plot=["pf6", "fec", "ea", "fea", "eaf", "feaf"],
            x_label="Species",
            y_label="Pairing",
            title="Graph",
        )


def test_compare_solvent_dicts_sensitivity(eax_solutes):
    # solvent names are case-sensitive, so names in solvents_to_plot and rename_solvent_dict should be consistent
    # this test should handle an exception
    with pytest.raises(Exception):
        fig = compare_pairing(
            eax_solutes,
            rename_solvent_dict={
                "EA": "EAx",
                "fEA": "EAx",
                "EAf": "EAx",
                "fEAf": "EAx",
            },
            solvents_to_plot=["PF6", "FEC", "EAx"],
            x_label="Species",
            y_label="Pairing",
            title="Graph",
        )


# compare_pairing tests
def test_compare_pairing_default_eax(eax_solutes):
    # call compare_pairing with only one required argument
    # also tests how the code handles eax systems
    fig = compare_pairing(eax_solutes)
    assert len(fig.data) == 4
    # fig.show()


def test_compare_pairing_case1(eax_solutes):
    # solvents_to_plot on x axis, each bar is a solute
    fig = compare_pairing(
        eax_solutes,
        solvents_to_plot=["fec", "pf6"],
        x_label="Species",
        y_label="Pairing",
        title="Bar Graph of Solvent Pairing",
    )
    assert len(fig.data) == 4
    for bar in fig.data:
        assert set(bar.x) == {"fec", "pf6"}
    # fig.show()


def test_compare_pairing_case2(eax_solutes):
    # solutes on x axis, each bar is an element of solvents_to_plot
    fig = compare_pairing(
        eax_solutes,
        solvents_to_plot=["pf6", "fec"],
        x_label="Solute",
        y_label="Pairing",
        title="Bar Graph of Solvent Pairing",
        x_axis="solute",
    )
    assert len(fig.data) == 2
    for bar in fig.data:
        assert set(bar.x) == {"feaf", "eaf", "fea", "ea"}
    # fig.show()


def test_compare_pairing_case3(eax_solutes):
    # solvents_to_plot on x axis, each line is a solute
    fig = compare_pairing(
        eax_solutes,
        solvents_to_plot=["pf6", "fec"],
        x_label="Solute",
        y_label="Pairing",
        title="Line Graph of Solvent Pairing",
        series=True,
    )
    assert len(fig.data) == 4
    for line in fig.data:
        assert set(line.x) == {"fec", "pf6"}
    # fig.show()


def test_compare_pairing_case4(eax_solutes):
    # solutes on x axis, each line is an element of solvents_to_plot
    fig = compare_pairing(
        eax_solutes,
        solvents_to_plot=["pf6", "fec"],
        x_label="Solute",
        y_label="Pairing",
        title="Line Graph of Solvent Pairing",
        x_axis="solute",
        series=True,
    )
    assert len(fig.data) == 2
    for line in fig.data:
        assert set(line.x) == {"feaf", "eaf", "fea", "ea"}
    # fig.show()


def test_compare_pairing_switch_solvents_to_plot_order(eax_solutes):
    # same test as test_compare_pairing_case4, except order for solvents_to_plot is switched
    fig = compare_pairing(
        eax_solutes,
        solvents_to_plot=["fec", "pf6"],
        x_label="Solute",
        y_label="Pairing",
        title="Line Graph of Solvent Pairing",
        x_axis="solute",
        series=True,
    )
    assert len(fig.data) == 2
    for line in fig.data:
        assert set(line.x) == {"feaf", "eaf", "fea", "ea"}
    # fig.show()


def test_compare_pairing_rename_solvent_dict(eax_solutes):
    # rename solvent names into the generic "EAx" form
    fig = compare_pairing(
        eax_solutes,
        rename_solvent_dict={"ea": "EAx", "fea": "EAx", "eaf": "EAx", "feaf": "EAx"},
        solvents_to_plot=["pf6", "fec", "EAx"],
        x_label="Species",
        y_label="Pairing",
        title="Bar Graph of Solvent Pairing",
    )
    assert len(fig.data) == 4
    for bar in fig.data:
        assert set(bar.x) == {"pf6", "fec", "EAx"}
    # fig.show()


def test_compare_free_solvents(eax_solutes):
    compare_free_solvents(eax_solutes, solvents_to_plot=["fec", "pf6"])


def test_compare_diluent(eax_solutes):
    compare_diluent(eax_solutes, solvents_to_plot=["fec", "pf6"])


# compare_coordination_numbers tests
def test_compare_coordination_numbers_default_eax(eax_solutes):
    # call compare_coordination_numbers with only one required argument
    # also tests how the code handles eax systems
    fig = compare_coordination_numbers(eax_solutes)
    assert len(fig.data) == 4
    # fig.show()


def test_compare_coordination_numbers_solute_four_cases(eax_solutes):
    fig = compare_coordination_numbers(eax_solutes, x_axis='solute')
    assert len(fig.data) == 6

    fig = compare_coordination_numbers(eax_solutes, x_axis='solute', series=True)
    assert len(fig.data) == 6

    rename = {
        "ea": "EAx",
        "fea": "EAx",
        "eaf": "EAx",
        "feaf": "EAx",
    }
    fig = compare_coordination_numbers(eax_solutes, x_axis='solute', rename_solvent_dict=rename)
    assert len(fig.data) == 3

    fig = compare_coordination_numbers(eax_solutes, x_axis='solute', series=True, rename_solvent_dict=rename)
    assert len(fig.data) == 3

    fig = compare_coordination_numbers(
        eax_solutes,
        x_axis='solute',
        rename_solvent_dict=rename,
        series=True,
        solvents_to_plot=['EAx', 'pf6'],
    )
    assert len(fig.data) == 2


def test_compare_coordination_numbers_case1(eax_solutes):
    # solvents_to_plot on x axis, each bar is a solute
    fig = compare_coordination_numbers(
        eax_solutes,
        solvents_to_plot=["fec", "pf6"],
        x_label="Species",
        y_label="Coordination",
        title="Bar Graph of Coordination Numbers",
    )
    assert len(fig.data) == 4
    for bar in fig.data:
        assert set(bar.x) == {"fec", "pf6"}
    # fig.show()


def test_compare_coordination_numbers_case2(eax_solutes):
    # solutes on x axis, each bar is an element of solvents_to_plot
    fig = compare_coordination_numbers(
        eax_solutes,
        solvents_to_plot=["pf6", "fec"],
        x_label="solute",
        y_label="Coordination",
        title="Bar Graph of Coordination Numbers",
        x_axis="solute",
    )
    assert len(fig.data) == 2
    for bar in fig.data:
        assert set(bar.x) == {"feaf", "eaf", "fea", "ea"}
    # fig.show()


def test_compare_coordination_numbers_case3(eax_solutes):
    # solvents_to_plot on x axis, each line is a solute
    fig = compare_coordination_numbers(
        eax_solutes,
        solvents_to_plot=["pf6", "fec"],
        x_label="solute",
        y_label="Coordination",
        title="Line Graph of Coordination Numbers",
        series=True,
    )
    assert len(fig.data) == 4
    for line in fig.data:
        assert set(line.x) == {"fec", "pf6"}
    # fig.show()


def test_compare_coordination_numbers_case4(eax_solutes):
    # solutes on x axis, each line is an element of solvents_to_plot
    fig = compare_coordination_numbers(
        eax_solutes,
        solvents_to_plot=["pf6", "fec"],
        x_label="solute",
        y_label="Coordination",
        title="Line Graph of Coordination Numbers",
        x_axis="solute",
        series=True,
    )
    assert len(fig.data) == 2
    for line in fig.data:
        assert set(line.x) == {"feaf", "eaf", "fea", "ea"}
    # fig.show()


# compare_residence_times tests
def test_compare_residence_times(eax_solutes):
    # this test should handle an exception relating to the acceptable arguments for res_type
    for solute in eax_solutes.values():
        residence = Residence.from_solute(solute)
        solute.residence = residence

    # with pytest.raises(Exception):
    compare_residence_times_cutoff(eax_solutes, solvents_to_plot=["fec", "pf6"])
    compare_residence_times_fit(eax_solutes, solvents_to_plot=["fec", "pf6"])


def test_compare_generic(eax_solutes):
    compare = _compare_function_generator(
        "pairing", "solvent_pairing", "hello", "This is a function"
    )
    fig = compare(
        eax_solutes,
        rename_solvent_dict={"ea": "EAx", "fea": "EAx", "eaf": "EAx", "feaf": "EAx"},
        solvents_to_plot=["pf6", "fec", "EAx"],
        x_label="Species",
        y_label="Pairing",
        title="Bar Graph of Solvent Pairing",
    )
    assert len(fig.data) == 4
    for bar in fig.data:
        assert set(bar.x) == {"pf6", "fec", "EAx"}
    # fig.show()


def test_plot_co_occurrence(solvation_data):
    speciation = Speciation(solvation_data, 10, 49)
    fig = plot_co_occurrence(speciation)
    # fig.show()
