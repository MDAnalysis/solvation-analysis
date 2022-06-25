import numpy as np
import pytest

from solvation_analysis.plotting import (
    plot_network_size_histogram,
    plot_shell_size_histogram,
)


def test_plot_network_size_histogram(networking):
    fig = plot_network_size_histogram(networking)
    fig.show()
    assert True

def test_plot_shell_size_histogram(run_solution):
    fig = plot_shell_size_histogram(run_solution)
    fig.show()
    assert True
