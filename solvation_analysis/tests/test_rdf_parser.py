import MDAnalysis as mda
import numpy as np
import matplotlib.pyplot as plt
import pytest
from solvation_analysis.rdf_parser import (
    quick_plot,
    identify_minima,
)


@pytest.mark.parametrize(
    'rdf_tag, minima',
    [('fec_F', 3),
     ('fec_O', 3),
     ('fec_all', 3)]
)
def test_identify_minima(rdf_tag, minima, rdf_bins, rdf_data):
    bins = rdf_bins[rdf_tag]
    rdf = rdf_data[rdf_tag]
    quick_plot(bins, rdf)
    return

