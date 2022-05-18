
import plotly
import plotly.graph_objects as go
import plotly.express as px



# single solution


def plot_speciation(solution):
    # square area
    return


def plot_coordination_number(solution):
    return


def plot_co_occurrence(solution):
    return


def plot_clustering(solution):
    # not in this branch yet
    return


def plot_coordinating_atoms(solution):
    return


# multiple solution


def compare_solvent_dicts(solutions, series=True, ignore_solvents=None):
    # generalist plotter, this can plot either bar or line charts of the
    # same data
    return 1


def compare_free_solvents(solutions):
    # this should be a grouped vertical bar chart or a line chart
    # 1.0 should be marked and annotated with a dotted line
    fig = compare_solvent_dicts()
    return


def compare_pairing(solutions):
    # this should be a grouped vertical bar chart or a line chart
    # 1.0 should be marked and annotated with a dotted line
    fig = compare_solvent_dicts()
    return


def compare_coordination_numbers(solutions):
    # this should be a stacked bar chart
    fig = compare_solvent_dicts()
    return

def compare_coordination_to_random(solutions):
    # this should compare the actual coordination numbers relative to a
    # statistically determined "random" distribution, ignoring sterics
    return


def compare_residence_time(solutions):
    # not in this branch yet
    return


def compare_speciation(solutions, series=True):
    # stacked bars
    # or square areas?
    return


def compare_rdfs(solutions, atoms):
    # can atom groups be matched to solutions / universes behind the scenes?
    # yes we can use atom.u is universe
    return



