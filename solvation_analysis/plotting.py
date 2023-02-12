"""
========
Plotting
========
:Authors: Orion Cohen and Lauren Lee
:Year: 2023
:Copyright: GNU Public License v3

The plotting functions are a convenient way to visualize data by taking solutions
as their input and generating a Plotly.Figure object.
"""

import plotly
import plotly.graph_objects as go
import plotly.express as px
import matplotlib
from copy import deepcopy

import numpy as np
import pandas as pd


# single solution
def plot_network_size_histogram(networking):
    """
    Returns a histogram of network sizes.

    Parameters
    ----------
    networking : Networking

    Returns
    -------
    fig : Plotly.Figure

    """
    network_sizes = networking.network_sizes
    sums = network_sizes.sum(axis=0)
    total_networks = sums.sum()
    fig = go.Figure()
    fig.add_trace(go.Bar(x=sums.index, y=sums.values / total_networks))
    fig.update_layout(xaxis_title_text="Network Size", yaxis_title_text="Fraction of All Networks",
                      title="Histogram of Network Sizes")
    fig.update_xaxes(type="category")
    return fig


def plot_shell_size_histogram(solution):
    """
    Returns a histogram of shell sizes.

    Parameters
    ----------
    solution : Solution

    Returns
    -------
    fig : Plotly.Figure

    """
    speciation_data = solution.speciation.speciation_data
    speciation_data["total"] = speciation_data.sum(axis=1)
    sums = speciation_data.groupby("total").sum()
    fig = go.Figure()
    totals = sums.T.sum()
    for column in sums.columns:
        fig.add_trace(go.Bar(x=sums.index.values, y=sums[column].values / totals, name=column))
    fig.update_layout(xaxis_title_text="Shell Size", yaxis_title_text="Fraction of Total Molecules",
                      title="Fraction of Solvents in Shells of Different Sizes")
    fig.update_xaxes(type="category")
    return fig


def compare_solvent_dicts(property_dict, rename_solvent_dict, solvents_to_plot, legend_label, x_axis="solvent",
                          series=False):
    """
    A generic plotting function that can compare dictionary data between multiple solutes.

    Parameters
    ----------
    property_dict : dict of {str: dict}
        a dictionary with the solution name as keys and a dict of {str: float} as values, where each key
        is the name of the solvent of each solution and each value is the property of interest
    rename_solvent_dict : dict of {str: str}
        Renames solvents within the plot, useful for comparing similar solvents in different solutes.
        The keys are the original solvent names and the values are the new name
        e.g. {"EAf" : "EAx", "fEAf" : "EAx"}
    solvents_to_plot : List[str]
        List of solvent names to be plotted, they will be plotted in given order.
        The solvents must be common to all systems in question. Renaming in `rename_solvent_dicts`
        is applied first, so the solvent names in `solvents_to_plot should match the `values` of that dict.
    legend_label : str
        title of legend as a string
    x_axis : str
        the value must be "solvent" or "solute" and decides which to plot the x_axis
    series : bool
        False for a bar graph, True for a line graph

    Returns
    -------
    fig : Plotly.Figure

    """
    property_dict = deepcopy(property_dict)
    # coerce solutions to a common name
    for solution_name in rename_solvent_dict:
        if solution_name in property_dict:
            common_name = rename_solvent_dict[solution_name]
            # remove the solution name from the properties dict and rename to the common name
            solution_property_value = property_dict[solution_name].pop(solution_name)
            property_dict[solution_name][common_name] = solution_property_value

    # filter out components of solution to only include those in solvents_to_plot
    if solvents_to_plot:
        all_solvents = [set(solution_dict.keys()) for solution_dict in property_dict.values()]
        valid_solvents = set.intersection(*all_solvents)
        invalid_solvents = set.union(*all_solvents) - valid_solvents
        if not set(solvents_to_plot).issubset(valid_solvents):
            raise Exception(
                f"solvents_to_plot must only include solvents that are"
                f"present in all solutes. Valid values are {valid_solvents}."
            )
        for solution_dict in property_dict.values():
            for solvent in invalid_solvents:
                solution_dict.pop(solvent, None)

    # generate figure and make a DataFrame of the data
    fig = go.Figure()
    df = pd.DataFrame(data=property_dict.values())
    df.index = list(property_dict.keys())

    if series and x_axis == "solvent":
        # each solution is a line
        df = df.transpose()
        fig = px.line(df, x=df.index, y=df.columns, labels={"variable": legend_label})
        fig.update_xaxes(type="category")
    elif series and x_axis == "solute":
        # each solvent is a line
        fig = px.line(df, x=df.index, y=df.columns, labels={"variable": legend_label})
        fig.update_xaxes(type="category")
    elif not series and x_axis == "solvent":
        # each solution is a bar
        df = df.transpose()
        fig = px.bar(df, x=df.index, y=df.columns, barmode="group", labels={"variable": legend_label})
    elif not series and x_axis == "solute":
        # each solvent is a bar
        fig = px.bar(df, x=df.index, y=df.columns, barmode="group", labels={"variable": legend_label})
    return fig


def compare_free_solvents(solutions):
    # this should be a grouped vertical bar chart or a line chart
    # 1.0 should be marked and annotated with a dotted line
    fig = compare_solvent_dicts()
    return


def compare_pairing(solutions, rename_solvent_dict=None, solvents_to_plot=None, x_label="Solvent", y_label="Pairing",
                    title="Graph of Pairing Data", legend_label="Legend", **kwargs):
    # this should be a grouped vertical bar chart or a line chart
    # 1.0 should be marked and annotated with a dotted line
    """
    Compares the pairing of multiple solutions.
    Parameters
    ----------
    solutions : dict of {str: Solution}, where the key is the name of the Solution object and the values are
        the Solution object
    rename_solvent_dict : dict of {str: str}, where the keys are strings of solvent names and the values are
        strings of a more generic name for the solvent (i.e. {"EAf" : "EAx", "fEAf" : "EAx"})
    solvents_to_plot : list of strings of solvent names that are common to all systems in question,
        graphed in the order that is passed into the function. Names of solutions are first swapped with the
        more generic name, if rename_solvent_dict is specified, before filtering for solution names
        specified by solvents_to_plot. In order for solvents_to_plot to execute properly, any solution names
        affected by the swap with rename_solvent_dict must be referenced by the generic name in solvents_to_plot.
    x_label : name of the x-axis as a string
    y_label : name of the y-axis as a string
    title : title of figure as a string
    legend_label : title of legend as a string
    kwargs : consists of the x_axis and series parameters
        x_axis : a string specifying "solvent" or "solution" to be graphed on the x_axis
        series : Boolean (False for a bar graph; True for a line graph)

    Returns
    -------
    fig : Plotly.Figure

    """
    rename_solvent_dict = rename_solvent_dict or {}
    pairing = {solution_name: solutions[solution_name].pairing.pairing_dict for solution_name in solutions}
    fig = compare_solvent_dicts(pairing, rename_solvent_dict, solvents_to_plot, legend_label, **kwargs)
    fig.update_layout(xaxis_title_text=x_label.title(), yaxis_title_text=y_label.title(), title=title.title())
    return fig


def compare_coordination_numbers(solutions, rename_solvent_dict=None, solvents_to_plot=None, x_label="Solvent",
                                 y_label="Coordination", title="Graph of Coordination Data", legend_label="Legend",
                                 **kwargs):
    """
    Compares the coordination numbers of multiple solutions.

    Parameters
    ----------
    solutions : dict of {str: Solution}, where the key is the name of the Solution object and the values are
        the Solution object
    rename_solvent_dict : dict of {str: str}, where the keys are strings of solvent names and the values are
        strings of a more generic name for the solvent (i.e. {"EAf" : "EAx", "fEAf" : "EAx"})
    solvents_to_plot : list of strings of solvent names that are common to all systems in question,
        graphed in the order that is passed into the function. Names of solutions are first swapped with the
        more generic name, if rename_solvent_dict is specified, before filtering for solution names
        specified by solvents_to_plot. In order for solvents_to_plot to execute properly, any solution names
        affected by the swap with rename_solvent_dict must be referenced by the generic name in solvents_to_plot.
    x_label : name of the x-axis as a string
    y_label : name of the y-axis as a string
    title : title of figure as a string
    legend_label : title of legend as a string
    kwargs : consists of the x_axis and series parameters
        x_axis : a string specifying "solvent" or "solution" to be graphed on the x_axis
        series : Boolean (False for a bar graph; True for a line graph)

    Returns
    -------
    fig : Plotly.Figure

    """
    rename_solvent_dict = rename_solvent_dict or {}
    coordination = {solution_name: solutions[solution_name].coordination.cn_dict for solution_name in solutions}
    fig = compare_solvent_dicts(coordination, rename_solvent_dict, solvents_to_plot, legend_label, **kwargs)
    fig.update_layout(xaxis_title_text=x_label.title(), yaxis_title_text=y_label.title(), title=title.title())
    return fig


def compare_residence_times(solutions, res_type="residence_times_fit", rename_solvent_dict=None, solvents_to_plot=None,
                            x_label="Solvent", y_label="Residence Time", title="Graph of Residence Time Data",
                            legend_label="Legend", **kwargs):
    """
    Compares the residence times of multiple solutions.

    Parameters
    ----------
    solutions : dict of {str: Solution}, where the key is the name of the Solution object and the values are
        the Solution object
    res_type : a string that is either "residence_times" or residence_times_fit"
    rename_solvent_dict : dict of {str: str}, where the keys are strings of solvent names and the values are
        strings of a more generic name for the solvent (i.e. {"EAf" : "EAx", "fEAf" : "EAx"})
    solvents_to_plot : list of strings of solvent names that are common to all systems in question,
        graphed in the order that is passed into the function. Names of solutions are first swapped with the
        more generic name, if rename_solvent_dict is specified, before filtering for solution names
        specified by solvents_to_plot. In order for solvents_to_plot to execute properly, any solution names
        affected by the swap with rename_solvent_dict must be referenced by the generic name in solvents_to_plot.
    x_label : name of the x-axis as a string
    y_label : name of the y-axis as a string
    title : title of figure as a string
    legend_label : title of legend as a string
    kwargs : consists of the x_axis and series parameters
        x_axis : a string specifying "solvent" or "solution" to be graphed on the x_axis
        series : Boolean (False for a bar graph; True for a line graph)

    Returns
    -------
    fig : Plotly.Figure

    """
    #
    if res_type not in ["residence_times", "residence_times_fit"]:
        raise ValueError("res_type must be either \"residence_times\" or \"residence_times_fit\"")

    res_time = {}
    for solution_name, solution in solutions.items():
        if not hasattr(solution, "residence"):
            raise ValueError("Solution's Residence analysis class is not instantiated.")
        res_time[solution_name] = getattr(solution.residence, res_type)

    rename_solvent_dict = rename_solvent_dict or {}
    fig = compare_solvent_dicts(res_time, rename_solvent_dict, solvents_to_plot, legend_label, **kwargs)
    fig.update_layout(xaxis_title_text=x_label.title(), yaxis_title_text=y_label.title(), title=title.title())
    return fig


# TODO: work on rdfs; make them tiled
# this will have to be implemented post-merge
# use iba_small_solutes (will return a solute that has three atom solutes
# solvents are on one axis and solutions are on the other
def compare_rdfs(solutions, atoms):
    # can atom groups be matched to solutions / universes behind the scenes?
    # yes we can use atom.u is universe
    return