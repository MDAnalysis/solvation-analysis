import numpy as np
from string import ascii_lowercase

from solvation_analysis.plotting import square_area

def test_square_area():
    data = np.array([50, 30, 10, 4, 4, 4, 3, 1, 1, 1, 1, 1])
    labels = np.array([char for char in ascii_lowercase[0: len(data)]], dtype='<U100')
    square_area(data, labels)