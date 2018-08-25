"""
This module contains functions that are commonly used across multiple learning algorithms
"""
import numpy as np


def predict(x, theta):
    """Predicts the output (y) given an matrix of x parameters.
    The hypothesis function for the model.
    """
    return np.dot(x, theta)
