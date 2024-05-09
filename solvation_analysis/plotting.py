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

from typing import Union, Optional, Any, Callable
from copy import deepcopy

import plotly
from plotly.subplots import make_subplots
import plotly.graph_objects as go
import plotly.express as px

import pandas as pd

from solvation_analysis.solute import Solute
from solvation_analysis.networking import Networking
from solvation_analysis.speciation import Speciation


# single solution
def plot_network_size_histogram(networking: Union[Networking, Solute]) -> go.Figure:
    """
    Plot a histogram of network sizes.

    Parameters
    ----------
    networking : Networking | Solution

    Returns
    -------
    fig : Plotly.Figure

    """
    if isinstance(networking, Solute):
        if not hasattr(networking, "networking"):
            raise ValueError("Solute networking analysis class must be instantiated.")
        networking = networking.networking
    network_sizes = networking.network_sizes
    sums = network_sizes.sum(axis=0)
    total_networks = sums.sum()
    fig = go.Figure()
    fig.add_trace(go.Bar(x=sums.index, y=sums.values / total_networks))
    fig.update_layout(
        xaxis_title_text="Network Size",
        yaxis_title_text="Fraction of All Networks",
        title="Histogram of Network Sizes",
        template="plotly_white",
    )
    fig.update_xaxes(type="category")
    return fig


def plot_shell_composition_by_size(speciation: Union[Speciation, Solute]) -> go.Figure:
    """
    Plot the composition of shells broken down by shell size.

    Parameters
    ----------
    speciation : Speciation | Solution

    Returns
    -------
    fig : Plotly.Figure

    """
    if isinstance(speciation, Solute):
        if not hasattr(speciation, "speciation"):
            raise ValueError("Solute speciation analysis class must be instantiated.")
        speciation = speciation.speciation
    speciation_data = speciation.speciation_data.copy()
    speciation_data["total"] = speciation_data.sum(axis=1)
    sums = speciation_data.groupby("total").sum()
    fig = go.Figure()
    totals = sums.T.sum()
    for column in sums.columns:
        fig.add_trace(
            go.Bar(x=sums.index.values, y=sums[column].values / totals, name=column)
        )
    fig.update_layout(
        xaxis_title_text="Shell Size",
        yaxis_title_text="Fraction of Total Molecules",
        title="Fraction of Solvents in Shells of Different Sizes",
        template="plotly_white",
    )
    fig.update_xaxes(type="category")
    return fig


def plot_co_occurrence(
    speciation: Union[Speciation, Solute], colorscale: Optional[Any] = None
) -> go.Figure:
    """
    Plot the co-occurrence matrix of the solute using Plotly.

    Co-occurrence represents the extent to which solvents occur with each other
    relative to random. Values higher than 1 mean that solvents occur together
    more often than random and values lower than 1 mean solvents occur together
    less often than random. "Random" is calculated based on the total number of
    solvents participating in solvation, it ignores solvents in the diluent.

    Args
    ----
    speciation: Speciation | Solution
    colorscale : any valid argument to Plotly colorscale.

    Returns
    -------
    fig : plotly.graph_objs.Figure
    """
    if isinstance(speciation, Solute):
        if not hasattr(speciation, "speciation"):
            raise ValueError("Solute speciation analysis class must be instantiated.")
        speciation = speciation.speciation

    solvent_names = speciation.speciation_data.columns.values

    if colorscale:
        colorscale = colorscale
    else:
        min_val = speciation.solvent_co_occurrence.min().min()
        max_val = speciation.solvent_co_occurrence.max().max()
        range_val = max_val - min_val

        colorscale = [
            [0, "rgb(67,147,195)"],
            [(1 - min_val) / range_val, "white"],
            [1, "rgb(214,96,77)"],
        ]

    # Create a heatmap trace with text annotations
    trace = go.Heatmap(
        x=solvent_names,
        y=solvent_names[::-1],  # Reverse the order of the y-axis labels
        z=speciation.solvent_co_occurrence.values,  # Keep the data in the original order
        text=speciation.solvent_co_occurrence.round(2).to_numpy(dtype=str),
        # Keep the text annotations in the original order
        hoverinfo="none",
        colorscale=colorscale,
    )

    # Update layout to display tick labels and text annotations
    layout = go.Layout(
        title="Solvent Co-Occurrence Matrix",
        xaxis=dict(
            tickmode="array",
            tickvals=list(range(len(solvent_names))),
            ticktext=solvent_names,
            tickangle=-30,
            side="top",
        ),
        yaxis=dict(
            tickmode="array",
            tickvals=list(range(len(solvent_names))),
            ticktext=solvent_names,
            autorange="reversed",
        ),
        margin=dict(l=60, r=60, b=60, t=100, pad=4),
        annotations=[
            dict(
                x=i,
                y=j,
                text=str(round(speciation.solvent_co_occurrence.iloc[j, i], 2)),
                font=dict(size=14, color="black"),
                showarrow=False,
            )
            for i in range(len(solvent_names))
            for j in range(len(solvent_names))
        ],
    )

    # Create and return the Figure object
    fig = go.Figure(data=[trace], layout=layout)
    return fig


def _make_rectangle(x: float, y: float, color: str) -> dict:
    """
    Create a rectangle shape for Plotly.

    Parameters
    ----------
    x : float
        The x-coordinate of the center of the rectangle.
    y : float
        The y-coordinate of the center of the rectangle.
    color : str
        The color of the rectangle.

    Returns
    -------
    go.layout.Shape
        The rectangle shape for Plotly.
    """

    x0 = x - 0.18
    y0 = y - 0.43
    x1 = x + 0.18
    y1 = y + 0.43
    h = 0.09
    rounded_bottom_left = f" M {x0 + h}, {y0} Q {x0}, {y0} {x0}, {y0 + h}"  #
    rounded_top_left = f" L {x0}, {y1 - h} Q {x0}, {y1} {x0 + h}, {y1}"
    rounded_top_right = f" L {x1 - h}, {y1} Q {x1}, {y1} {x1}, {y1 - h}"
    rounded_bottom_right = f" L {x1}, {y0 + h} Q {x1}, {y0} {x1 - h}, {y0}Z"
    path = (
        rounded_bottom_left
        + rounded_top_left
        + rounded_top_right
        + rounded_bottom_right
    )

    return dict(
        type="path",
        path=path,
        line=dict(color=color, width=2),
        fillcolor=color,
        layer="between",
    )


def _get_shell_name(row):
    result = []
    for column, value in row.items():
        result.append(f"{column} {value}")
    return "<br>".join(result)


def plot_speciation(
    speciation: Union[Speciation, Solute], shells: int = 6
) -> go.Figure:
    """
    Plot the solvation shell composition and fraction for the top solvation shells.

    Parameters
    ----------
    speciation : Speciation or Solute
        The Speciation or Solute object containing the speciation data.
    shells : int, optional
        The number of top solvation shells to plot. Default is 6.

    Returns
    -------
    fig : plotly.graph_objs.Figure
        The plot of the solvation shell composition and fraction.
    """
    if isinstance(speciation, Solute):
        if not hasattr(speciation, "speciation"):
            raise ValueError("Solute speciation analysis class must be instantiated.")
        speciation = speciation.speciation

    # Extract relevant data
    df = speciation.speciation_fraction.head(shells)
    fraction_data = df["fraction"]
    df = df.drop("fraction", axis=1)

    # Get unique solvents and assign colors
    solvents = df.columns.tolist()  # List of solvents
    colors = px.colors.qualitative.Plotly  # Get a list of Plotly's qualitative colors

    # If there are more solvents than colors, cycle through the colors again
    if len(solvents) > len(colors):
        colors = colors * (
            len(solvents) // len(colors) + 1
        )  # Repeat color list as needed
    color_map = dict(zip(solvents, colors))  # Create a color map for solvents

    # Prepare data for the plot
    x_vals = []
    y_vals = []
    solvent_names = []
    marker_colors = []  # To store color for each marker
    shell_names = []

    # Process each row to create stacks of points
    for index, row in df.iterrows():
        shell_names.append(_get_shell_name(row))
        total_count = 0
        for solvent, count in row.items():
            for i in range(count):
                x_vals.append(index)
                y_vals.append(
                    0.5 + i + total_count
                )  # Place each solvent count at different y-levels
                solvent_names.append(solvent)
                marker_colors.append(
                    color_map[solvent]
                )  # Use the dynamically assigned color
            total_count += count

    # Create scatter plot of solvent squares,
    trace1 = go.Scatter(
        x=x_vals,
        y=y_vals,
        mode="markers",
        marker=dict(size=25, color=marker_colors, opacity=0),  # Apply colors to markers
        text=solvent_names,
        hoverinfo="text",
        name="Solvents",
        legendgroup="solvents",
        showlegend=False,
    )

    trace2 = go.Scatter(
        x=df.index,
        y=fraction_data,
        mode="lines+markers",
        name="Fraction",
        yaxis="y2",
        line=dict(color="black"),
    )

    # Create the figure with two traces
    fig = go.Figure(data=[trace1, trace2])

    # Add traces for each solvent to create a legend
    for solvent, color in color_map.items():
        fig.add_trace(
            go.Scatter(
                x=[None],
                y=[None],
                mode="markers",
                marker=dict(size=10, color=color),
                name=solvent,
                legendgroup="solvents",
                showlegend=True,
            )
        )

    # Add squares with rounded corners on top of the points using the shapes API
    for x, y, color in zip(x_vals, y_vals, marker_colors):
        fig.add_shape(**_make_rectangle(x, y, color))

    # Update layout
    fig.update_layout(
        title="Top Solvation Shell Compositions",
        xaxis_title="Solvation Shell",
        # xaxis=dict(tickmode="linear", tick0=0, dtick=1),  # Set x-axis ticks to integers
        xaxis=dict(
            tickmode="array",
            tickvals=df.index,
            ticktext=shell_names,
        ),
        yaxis=dict(
            title="Shell Size",
            tickmode="array",
            tickvals=list(range(1, int(max(y_vals)) + 1)),
            range=[0, max(y_vals) + 1],  # Scale the top of the y-axis
            showgrid=False,
            side="right",
        ),
        yaxis2=dict(
            title="Shell Fraction",
            overlaying="y",
            side="left",
            range=[0, max(fraction_data) * 1.1],  # Scale the fraction axis
        ),
        template="plotly_white",
        margin=dict(l=20, r=20, t=60, b=20),  # Add padding to the edges of the plot
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1,
            xanchor="right",
            x=1,
        ),  # Add legend at the top
    )

    return fig


def plot_rdfs(
    solute: Solute,
    show_cutoff: bool = True,
    x_axis_solute: bool = False,
    merge_on_x: bool = False,
    merge_on_y: bool = False,
):
    """
    Plot the radial distribution functions (RDFs) of solute-solvent pairs.

    Parameters
    ----------
    solute : Solute
        The Solute object containing the RDF data.
    show_cutoff : bool, optional
        Whether to display the solvation radius cutoff lines. Default is True.
    x_axis_solute : bool, optional
        Whether to place the solute on the x-axis. Default is False.
    merge_on_x : bool, optional
        Whether to merge subplots along the x-axis. Default is False.
    merge_on_y : bool, optional
        Whether to merge subplots along the y-axis. Default is False.

    Returns
    -------
    fig : plotly.graph_objs.Figure
        The plot of the radial distribution functions.
    """
    # Determine the grid dimensions based on merge settings
    data = solute.rdf_data
    n_cols = 1 if merge_on_y else len(data)
    n_rows = 1 if merge_on_x else len(data[list(data.keys())[0]])

    x_title, y_title = "Solvent", "Solute"

    if x_axis_solute:
        n_rows, n_cols = n_cols, n_rows
        x_title, y_title = y_title, x_title

    # Create subplots
    fig = make_subplots(
        rows=n_rows,
        cols=n_cols,
        shared_xaxes=True,
        shared_yaxes=True,
        x_title=x_title,
        y_title=y_title,
    )

    # Create a color mapping dictionary
    color_map = {}
    colors = plotly.colors.qualitative.Plotly

    # Iterate over the data and add traces to the subplots
    for i, (key, values) in enumerate(data.items()):
        for j, (sub_key, sub_values) in enumerate(values.items()):
            x, y = sub_values
            col = i * (not merge_on_y) + 1
            row = j * (not merge_on_x) + 1

            if x_axis_solute:
                row, col = col, row

            # Assign a color to the sub-key if not already assigned
            if sub_key not in color_map:
                show_legend = True
                color_map[sub_key] = colors[len(color_map) % len(colors)]
            else:
                show_legend = False

            fig.add_trace(
                go.Scatter(
                    x=x,
                    y=y,
                    name=sub_key,
                    line=dict(color=color_map[sub_key]),
                    legendgroup=sub_key,
                    showlegend=show_legend,
                ),
                row=row,
                col=col,
            )
            fig.update_yaxes(title_text=key, row=row, col=1)
            fig.update_xaxes(title_text=sub_key, row=n_rows, col=col)

    # Update the layout
    fig.update_layout(
        title_text="Radial Distribution Functions of Solute-Solvent Pairs",
        template="plotly_white",
        margin=dict(
            l=100,
            b=80,
        ),
    )
    fig.update_annotations(x=0.5, y=-0.05, selector={"text": x_title})
    fig.update_annotations(y=0.5, x=-0.03, selector={"text": y_title})

    if not (merge_on_x or merge_on_y) and show_cutoff:
        for col, solute in enumerate(solute.atom_solutes.values()):
            for row, (solvent, radius) in enumerate(solute.radii.items()):
                if x_axis_solute:
                    row, col = col, row
                fig.add_vline(
                    x=radius,
                    row=row,
                    col=col,
                    label=dict(
                        text="solvation radius",
                        textposition="top center",
                        yanchor="top",
                    ),
                )

    return fig


def compare_networking(solutions, series=False):
    # valid_x_axis = set(["solvent", "solute"])
    # assert x_axis in valid_x_axis, "x_axis must be equal to 'solute' or 'solvent'."
    # x_label = x_label or x_axis
    # legend_label = legend_label or (valid_x_axis - {x_axis}).pop()

    property_dict = {}
    for solute_name, solute in solutions.items():
        if not hasattr(solute, "networking"):
            raise ValueError("Solute networking analysis class must be instantiated.")
        property_dict[solute_name] = solute.networking.solute_status

    solvents_to_plot = ["isolated", "paired", "networked"]

    fig = compare_solvent_dicts(
        property_dict=property_dict,
        rename_solvent_dict={},
        solvents_to_plot=solvents_to_plot,
        legend_label="Solute Status",
        x_axis_solute=True,
        series=series,
    )

    fig.update_layout(
        xaxis_title_text="Solute",
        yaxis_title_text="Solute Status Fraction",
        title="Fraction of Solutes Isolated, Paired, and Networked",
        template="plotly_white",
    )
    return fig


def compare_solvent_dicts(
    property_dict: dict[str, dict[str, float]],
    rename_solvent_dict: dict[str, str],
    solvents_to_plot: list[str],
    legend_label: str,
    x_axis_solute: str = False,
    series: bool = False,
) -> go.Figure:
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
    for solution_name in rename_solvent_dict.keys():
        if solution_name in property_dict:
            common_name = rename_solvent_dict[solution_name]
            # remove the solution name from the properties dict and rename to the common name
            solution_property_value = property_dict[solution_name].pop(solution_name)
            property_dict[solution_name][common_name] = solution_property_value

    # filter out components of solution to only include those in solvents_to_plot
    if solvents_to_plot:
        all_solvents = [
            set(solution_dict.keys()) for solution_dict in property_dict.values()
        ]
        valid_solvents = set.intersection(*all_solvents)
        if not set(solvents_to_plot).issubset(valid_solvents):
            raise Exception(
                f"solvents_to_plot must only include solvents that are "
                f"present in all solutes. Valid values are {valid_solvents}."
            )
        for solute_name, solution_dict in property_dict.items():
            new_solution_dict = {
                solvent: value
                for solvent, value in solution_dict.items()
                if solvent in solvents_to_plot
            }
            property_dict[solute_name] = new_solution_dict

    # generate figure and make a DataFrame of the data
    fig = go.Figure()
    df = pd.DataFrame(data=property_dict.values())
    df.index = list(property_dict.keys())

    if series and not x_axis_solute:
        # each solution is a line
        df = df.transpose()
        fig = px.line(
            df,
            x=df.index,
            y=df.columns,
            labels={"variable": legend_label},
            markers=True,
        )
        fig.update_xaxes(type="category")
    elif series and x_axis_solute:
        # each solvent is a line
        fig = px.line(df, y=df.columns, labels={"variable": legend_label}, markers=True)
        fig.update_xaxes(type="category")
    elif not series and not x_axis_solute:
        # each solution is a bar
        df = df.transpose()
        fig = px.bar(
            df,
            x=df.index,
            y=df.columns,
            barmode="group",
            labels={"variable": legend_label},
        )
    elif not series and x_axis_solute:
        # each solvent is a bar
        fig = px.bar(
            df,
            y=df.columns,
            barmode="group",
            labels={"variable": legend_label},
        )
    return fig


def _compare_function_generator(
    analysis_object: str,
    attribute: str,
    title: str,
    top_level_docstring: str,
) -> Callable:
    def compare_func(
        solutions,
        rename_solvent_dict=None,
        solvents_to_plot=None,
        x_axis_solute=False,
        series=False,
        title=title,
        x_label=None,
        y_label=attribute.replace("_", " ").title(),
        legend_label=None,
    ):
        x_axis = "solute" if x_axis_solute else "solvent"
        x_label = x_label or x_axis
        legend_label = legend_label or x_axis

        property = {}
        for solute_name, solute in solutions.items():
            if not hasattr(solute, analysis_object):
                raise ValueError(
                    f"Solute {analysis_object} analysis class must be instantiated."
                )
            property[solute_name] = getattr(getattr(solute, analysis_object), attribute)

        rename_solvent_dict = rename_solvent_dict or {}
        fig = compare_solvent_dicts(
            property,
            rename_solvent_dict,
            solvents_to_plot,
            legend_label.title(),
            x_axis,
            series,
        )
        fig.update_layout(
            xaxis_title_text=x_label.title(),
            yaxis_title_text=y_label.title(),
            title=title.title(),
            template="plotly_white",
        )
        return fig

    arguments_docstring = """

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
    x_axis : str
        the value must be "solvent" or "solute" and decides which to plot the x_axis
    series : bool
        False for a bar graph, True for a line graph
    x_label : str
        title of the x-axis
    y_label : str
        title of the y-axis
    title : str
        title of figure
    legend_label : str
        title of legend

    Returns
    -------
    fig : Plotly.Figure
    """
    compare_func.__doc__ = top_level_docstring + arguments_docstring
    return compare_func


compare_pairing = _compare_function_generator(
    "pairing",
    "solvent_pairing",
    "Fractional Pairing of Solvents",
    "Compare the solute-solvent pairing.",
)


compare_free_solvents = _compare_function_generator(
    "pairing",
    "fraction_free_solvents",
    "Free Solvents in Solutes",
    "Compare the relative fraction of free solvents.",
)


compare_diluent = _compare_function_generator(
    "pairing",
    "diluent_composition",
    "Diluent Composition of Solutes",
    "Compare the diluent composition.",
)


compare_coordination_numbers = _compare_function_generator(
    "coordination",
    "coordination_numbers",
    "Coordination Numbers of Solvents",
    "Compare the coordination numbers.",
)


compare_coordination_vs_random = _compare_function_generator(
    "coordination",
    "coordination_vs_random",
    "Coordination Compare to Random Distribution of Solvents",
    "Compare the coordination numbers.",
)

compare_residence_times_cutoff = _compare_function_generator(
    "residence",
    "residence_times_cutoff",
    "Solute-Solvent Residence Time",
    "Compare the solute-solvent residence times.",
)


compare_residence_times_fit = _compare_function_generator(
    "residence",
    "residence_times_fit",
    "Solute-Solvent Residence Time.",
    "Compare the solute-solvent residence times.",
)


# TODO: work on rdfs; make them tiled
# this will have to be implemented post-merge
# use iba_small_solutes (will return a solute that has three atom solutes
# solvents are on one axis and solutions are on the other
def compare_rdfs(solutions, atoms):
    # can atom groups be matched to solutions / universes behind the scenes?
    # yes we can use atom.u is universe
    return
