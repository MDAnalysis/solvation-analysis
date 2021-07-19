import matplotlib.pyplot as plt
import pytest

from copy import deepcopy

from solvation_analysis.analysis import Solute


@pytest.fixture
def default_solute(atom_groups):
    li = atom_groups['li']
    pf6 = atom_groups['pf6']
    bn = atom_groups['bn']
    fec = atom_groups['fec']
    return Solute(li, {'pf6': pf6, 'bn': bn, 'fec': fec})


@pytest.fixture
def prepared_solute(default_solute):
    default_solute.run_prepare()
    return default_solute  # TODO: will this work?


def test_plot_solvation_distance(rdf_bins_and_data_easy):
    bins, data = rdf_bins_and_data_easy['pf6_all']
    fig, ax = Solute._plot_solvation_radius(bins, data, 2)
    plt.show()  # TODO: comment for global testing


def test_run_prepare(default_solute):
    default_solute.run_prepare()
    return


def test_run(prepared_solute):
    prepared_solute.run()


