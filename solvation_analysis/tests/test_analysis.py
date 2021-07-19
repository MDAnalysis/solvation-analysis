import matplotlib.pyplot as plt
import pytest

from copy import deepcopy

from solvation_analysis.analysis import Solute
import numpy as np

@pytest.fixture
def default_solute(atom_groups):
    li = atom_groups['li']
    pf6 = atom_groups['pf6']
    bn = atom_groups['bn']
    fec = atom_groups['fec']
    return Solute(li, {'pf6': pf6, 'bn': bn, 'fec': fec}, radii={'pf6': 2.8})


@pytest.fixture
def prepared_solute(default_solute):
    default_solute.run_prepare()
    return default_solute  # TODO: will this work?


def test_plot_solvation_distance(rdf_bins_and_data_easy):
    bins, data = rdf_bins_and_data_easy['pf6_all']
    fig, ax = Solute._plot_solvation_radius(bins, data, 2)
    # plt.show()  # TODO: comment for global testing


def test_instantiate_solute(default_solute):
    assert len(default_solute.radii) == 1
    assert len(default_solute.rdf_data) == 0
    assert len(default_solute.rdf_plots) == 0
    assert callable(default_solute.kernel)
    assert default_solute.solute.n_residues == 49
    assert default_solute.solvents['pf6'].n_residues == 49
    assert default_solute.solvents['fec'].n_residues == 237
    assert default_solute.solvents['bn'].n_residues == 363


def test_run_prepare(default_solute):
    default_solute.run_prepare()
    assert len(default_solute.radii) == 3
    assert len(default_solute.rdf_data) == 3
    assert len(default_solute.rdf_plots) == 3
    assert 2 < default_solute.radii['pf6'] < 3
    assert 2 < default_solute.radii['fec'] < 3
    assert 2 < default_solute.radii['bn'] < 3
    # for fig, ax in default_solute.rdf_plots.values():
    #     plt.show()  # TODO: comment this out before push


def test_run(prepared_solute):
    prepared_solute.run()


