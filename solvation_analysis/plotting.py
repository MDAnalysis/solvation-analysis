
import plotly
import plotly.graph_objects as go
import plotly.express as px
import matplotlib

import numpy as np
import pandas as pd

# single solution

def plot_histogram(solution):
    # histogram of what?
    return

def format_graph(fig, title, x_axis, y_axis):
    # a formatting/styling function for each graph that deals with titles/labels
    # make a legend
    """

    Parameters
    ----------
    fig :
    title :
    x_axis :
    y_axis :

    Returns
    -------
    fig : Plotly.Figure

    """
    fig.update_layout(xaxis_title_text=x_axis.title(), yaxis_title_text=y_axis.title(), title=title.title())
    return fig

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
    fig.add_trace(go.Bar(x=sums.index, y=sums.values/total_networks))
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
    # TODO: orion suggests maybe consider having the option to replace the solvent names with
    # custom solvent names, perhaps via an input dictionary
    # ideally this same API could be reused throughout the plotting API

    speciation_data = solution.speciation.speciation_data
    speciation_data["total"] = speciation_data.sum(axis=1)
    sums = speciation_data.groupby("total").sum()
    fig = go.Figure()
    totals = sums.T.sum()
    for column in sums.columns:
        fig.add_trace(go.Bar(x=sums.index.values, y=sums[column].values/totals, name=column))
    fig.update_layout(xaxis_title_text="Shell Size", yaxis_title_text="Fraction of Total Molecules",
                      title="Fraction of Solvents in Shells of Different Sizes")
    fig.update_xaxes(type="category")
    return fig


def plot_speciation(solution):
    # square area
    # should be doable with plotly.express.imshow and go.add_annotations
    return


def plot_co_occurrence(solution):
    return


def plot_clustering(solution):
    # not in this branch yet
    return


def plot_coordinating_atoms(solution):
    # for each solvent
    # by atom type? could allow by element or other features?
    # bar chart with one bar for each solvent
    # normalized
    return


# multiple solution


def compare_solvent_dicts(properties, keep_solvents, x_axis="species", series=False):
    # generalist plotter, this can plot either bar or line charts of the same data
    # this function should be under the hood, users shouldn't have to know how it works
    """

    Parameters
    ----------
    properties : dictionary of the solvent property to be compared
    keep_solvents : a list of strings of solvent names that are common to all systems in question
    x_axis : a string specifying "species" or "solution" to be graphed on the x_axis
    series : Boolean (defaults to False for a bar graph; True specifies a line graph)
    kwargs : consists of the x_axis and series parameters

    Returns
    -------
    fig : Plotly.Figure (generic plot)

    """
    fig = go.Figure()

    cleaned_properties = properties
    if keep_solvents:
        cleaned_properties = clean(properties, keep_solvents)
    df = pd.DataFrame(data=cleaned_properties)

    if series and x_axis == "species":
        # each solution is a line
        df = df.transpose()
        fig = px.line(df, x=df.index, y=df.columns)
        fig.update_xaxes(type="category")
    elif series and x_axis == "solution":
        # each species is a line
        fig = px.line(df, x=df.index, y=df.columns)
        fig.update_xaxes(type="category")
    elif not series and x_axis == "species":
        # each solution is a bar
        df = df.transpose()
        fig = px.bar(df, x=df.index, y=df.columns, barmode="group")
    elif not series and x_axis == "solution":
        # each species is a bar
        fig = px.bar(df, x=df.index, y=df.columns, barmode="group")
    return fig


def clean(properties, keep_solvents):
    cleaned_properties = []
    for i in range(len(properties)):
        cleaned_properties.append({})
        for solvent in properties[i]:
            if solvent in keep_solvents:
                cleaned_properties[i][solvent] = properties[i][solvent]
    return cleaned_properties


def catch_different_solvents(keep_solvents, solutions):
    if keep_solvents:
        solvents_list = []
        for solution in solutions:
            temp_solvents = list(solutions[solution].solvents)
            solvents_list.append(list(set(temp_solvents).intersection(set(keep_solvents))))
    else:
        solvents_list = [list(solutions[solution].solvents.keys()) for solution in solutions]

    if not all(elem == solvents_list[0] for elem in solvents_list):
        raise ValueError("Solutions must have identical solvent compositions. Make sure that keep_solvents for each"
            " solution lists only the solvents common to all solutions")


def compare_free_solvents(solutions):
    # this should be a grouped vertical bar chart or a line chart
    # 1.0 should be marked and annotated with a dotted line
    fig = compare_solvent_dicts()
    return


def compare_pairing(solutions, keep_solvents=None, **kwargs):
    # this should be a grouped vertical bar chart or a line chart
    # 1.0 should be marked and annotated with a dotted line
    """
    Compares the pairing of multiple solutions.
    Parameters
    ----------
    solutions : a dictionary of Solution objects
    keep_solvents : a list of strings of solvent names that are common to all systems in question
    kwargs: consists of the x_axis and series parameters
        x_axis : a string specifying "species" or "solution" to be graphed on the x_axis
        series : Boolean (False for a bar graph; True for a line graph)

    Returns
    -------
    fig : Plotly.Figure

    """
    catch_different_solvents(keep_solvents, solutions)
    pairing = [solutions[solution].pairing.pairing_dict for solution in solutions]
    pairing_dict = {}
    for solution in solutions:
        pairing_dict[solution] = solutions[solution].pairing.pairing_dict
    return compare_solvent_dicts(pairing, keep_solvents, **kwargs)


def compare_coordination_numbers(solutions, keep_solvents=None, **kwargs):
    # this should be a stacked bar chart, horizontal?
    """
    Compares the coordination numbers of multiple solutions.

    Parameters
    ----------
    solutions : a list of Solution objects
    series : Boolean (False when a line chart is not wanted)
    ignore : list of strings of solvent names to ignore

    Returns
    -------
    fig : Plotly.Figure

    """
    keep_solvents = keep_solvents or []
    catch_different_solvents(keep_solvents, solutions)
    coordination = [solutions[solution].coordination.cn_dict for solution in solutions]
    coord_dict = {}
    for solution in solutions:
        coord_dict[solution] = solutions[solution].pairing.pairing_dict
    return compare_solvent_dicts(coordination, keep_solvents, **kwargs)


def compare_coordination_to_random(solutions):
    # this should compare the actual coordination numbers relative to a
    # statistically determined "random" distribution, ignoring sterics
    return


def compare_residence_times(solutions, series=False, ignore=None):
    # not in this branch yet
    # this should be a grouped vertical bar chart or a line chart
    """
    Compares the coordination numbers of multiple solutions.

    Parameters
    ----------
    solutions : a list of Solution objects
    series : Boolean
    ignore : list of strings of solvent names to be ignored

    Returns
    -------
    fig : Plotly.Figure

    """
    catch_different_solvents(solutions)
    residence = [solution.residence.residence_times for solution in solutions]
    residence = set(residence) - set(ignore)
    return compare_solvent_dicts(residence, series)


def compare_solute_status(solutions):
    # not in this branch yet
    # this should be a grouped vertical bar chart or a line chart
    fig = compare_solvent_dicts()
    return

def compare_speciation(solutions, series=True):
    # stacked bars, grouped or stacked
    # or square areas?

    return


def compare_rdfs(solutions, atoms):
    # can atom groups be matched to solutions / universes behind the scenes?
    # yes we can use atom.u is universe
    return



