
import plotly
import plotly.graph_objects as go
import plotly.express as px
import matplotlib

import numpy as np
import pandas as pd


def square_area(data, labels, cutoff=1):
    data = np.floor(100 * data / np.sum(data)).astype(int)
    do_exceed_cutoff = data > cutoff
    label_array_short = np.repeat(labels[do_exceed_cutoff], data[do_exceed_cutoff])
    # label_array = np.pad(
    #     label_array_short,
    #     (0, 100 - len(label_array_short)),
    #     'constant',
    #     constant_values=(np.nan, 'other'),
    # )
    colors = px.colors.qualitative.Dark24
    color_array_short = np.repeat(colors[:sum(do_exceed_cutoff)], data[do_exceed_cutoff])
    color_array = np.pad(
        color_array_short,
        (0, 100 - len(label_array_short)),
        'constant',
        constant_values=("000000", '#EEEEEE'),
    )
    rgb_array_unwrapped = [np.array(plotly.colors.hex_to_rgb(color)) for color in color_array]
    rgb_values = np.reshape(rgb_array_unwrapped, (10,10,3))
    fig = px.imshow(rgb_values, zmin=0, zmax=255)
    fig.show()
    return

# single solution

def plot_histogram(solution):
    # histogram of what?
    return

def format_graph(fig, title, x_axis, y_axis):
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
    fig = go.Figure()
    fig.add_trace(go.Bar(x=sums.index, y=sums.values))
    fig.update_layout(xaxis_title_text="Network Size", yaxis_title_text="Frequency", title="Histogram of Network Sizes")
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
    for column in sums.columns:
        fig.add_trace(go.Bar(x=sums.index.values, y=sums[column].values, name=column))
    fig.update_layout(xaxis_title_text="Shell Size", yaxis_title_text="Number of Molecules",
                      title="Histogram of Shell Sizes")
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


def compare_solvent_dicts(properties_dict, properties, x_axis, series, keep_solvents):
    # generalist plotter, this can plot either bar or line charts of the
    # same data
    # manages only properties and names of the plots
    # make a separate formatting/styling function for each graph that deals with titles/labels
    """

    Parameters
    ----------
    properties : dictionary of solvent properties
    name :
    series : Boolean (False when a line chart is not wanted)
    x_axis :
    keep_solvents :

    Returns
    -------
    fig : Plotly.Figure (generic plot)

    """
    fig = go.Figure()
    cleaned_properties = properties
    if keep_solvents:
        cleaned_properties = clean(properties, keep_solvents)

    if series:
        if x_axis == "species":
            # each solution is a line
            for property_dict in cleaned_properties:
                fig.add_trace(go.Scatter(x=list(property_dict.keys()), y=list(property_dict.values())))
        elif x_axis == "solution":
            # each species is a line
            for property_dict in cleaned_properties:
                fig.add_trace(go.Scatter(x=list(property_dict.values()), y=list(property_dict.keys())))
    else:
        if x_axis == "species":
            # each solution is a bar
            for property_dict in cleaned_properties:
                fig.add_trace(go.Bar(x=list(property_dict.keys()), y=list(property_dict.values())))
        elif x_axis == "solution":
            # each species is a bar
            df = pd.DataFrame(data=properties_dict)

    return fig

def clean(properties, keep_solvents):
    cleaned_properties = []
    for i in range(len(properties)):
        cleaned_properties.append({})
        for solvent in properties[i]:
            if solvent in keep_solvents:
                cleaned_properties[i][solvent] = properties[i][solvent]
    return cleaned_properties


def catch_different_solvents(solvents):
    if not all(elem == solvents[0] for elem in solvents):
        raise ValueError("Solutions must have identical solvent compositions. Make sure that keep_solvents for each"
            " solution lists only the solvents common to all solutions")

def compare_free_solvents(solutions):
    # this should be a grouped vertical bar chart or a line chart
    # 1.0 should be marked and annotated with a dotted line
    fig = compare_solvent_dicts()
    return


def compare_pairing(solutions, x_axis, series=False, keep_solvents=None):
    # this should be a grouped vertical bar chart or a line chart
    # *** should there be another (boolean) parameter that
    # 1.0 should be marked and annotated with a dotted line
    """
    Compares the pairing of multiple solutions.
    Parameters
    ----------
    solutions : a dict of Solution objects (should not be mutable)
    series : Boolean (False when a line chart is not wanted)
    keep_solvents : a list of solvent names (strings) to keep in the analysis

    Returns
    -------
    fig : Plotly.Figure

    """
    # you can also do kwargs instead of writing out all the keywords
    keep_solvents = keep_solvents or []
    if keep_solvents:
        solvents_list = []
        for solution in solutions:
            temp_solvents = list(solutions[solution].solvents)
            solvents_list.append(list(set(temp_solvents).intersection(set(keep_solvents))))
    else:
        solvents_list = [list(solutions[solution].solvents.keys()) for solution in solutions]

    catch_different_solvents(solvents_list)
    pairing = [solutions[solution].pairing.pairing_dict for solution in solutions]
    pairing_dict = {}
    for solution in solutions:
        pairing_dict[solution] = solutions[solution].pairing.pairing_dict
    return compare_solvent_dicts(pairing_dict, pairing, x_axis, series, keep_solvents)

def compare_coordination_numbers(solutions, series=False, ignore=None):
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
    # catch_different_solvents(solutions)
    coordination = [solution.coordination.cn_dict for solution in solutions]
    coordination = set(coordination) - set(ignore)
    return compare_solvent_dicts(coordination, series)

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



