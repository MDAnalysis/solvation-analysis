import pytest

from solvation_analysis.analysis_library import (
    _IonSpeciation,
    _CoordinationNumbers,
    _IonPairing,
)





def test_ion_speciation(solvation_results):
    speciation = _IonSpeciation(solvation_results)


def test_coordination_numbers(solvation_results):
    coordination = _CoordinationNumbers(solvation_results)


def test_ion_pairing(solvation_results):
    pairing = _IonPairing(solvation_results)
