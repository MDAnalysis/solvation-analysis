import numpy as np
import pytest

from solvation_analysis.speciation import Speciation


def test_speciation_from_solution(run_solution):
    speciation = Speciation.from_solution(run_solution)
    assert len(speciation.speciation_data) == 490


@pytest.mark.parametrize(
    "shell, percent",
    [
        ({'bn': 5, 'fec': 0, 'pf6': 0}, 0.357),
        ({'bn': 3, 'fec': 3, 'pf6': 0}, 0.004),
        ({'bn': 3, 'fec': 0, 'pf6': 1}, 0.016),
        ({'bn': 4}, 0.531),
    ],
)
def test_speciation_shell_percent(shell, percent, solvation_data):
    speciation = Speciation(solvation_data, 10, 49)
    percentage = speciation.shell_percent(shell)
    np.testing.assert_allclose(percent, percentage, atol=0.05)


@pytest.mark.parametrize(
    "shell, n_shells",
    [
        ({'bn': 5, 'fec': 0, 'pf6': 0}, 175),
        ({'bn': 3, 'fec': 3, 'pf6': 0}, 2),
        ({'bn': 3, 'fec': 0, 'pf6': 1}, 13),
        ({'bn': 4}, 260),
    ],
)
def test_speciation_find_shells(shell, n_shells, solvation_data):
    speciation = Speciation(solvation_data, 10, 49)
    df = speciation.find_shells(shell)
    assert len(df) == n_shells


@pytest.mark.parametrize(
    "solvent_one, solvent_two, correlation",
    [
        ('bn', 'bn', 0.98),
        ('fec', 'bn', 1.03),
        ('fec', 'pf6', 0.15),
    ],
)
def test_speciation_correlation(solvent_one, solvent_two, correlation, solvation_data):
    speciation = Speciation(solvation_data, 10, 49)
    df = speciation.co_occurrence
    np.testing.assert_allclose(df[solvent_one][solvent_two], correlation, atol=0.05)


def test_plot_correlation(solvation_data):
    speciation = Speciation(solvation_data, 10, 49)
    fig, ax = speciation.plot_co_occurrence()
    # fig.show()

