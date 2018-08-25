"""
This module contains functions that are commonly used across multiple learning algorithms
"""
import numpy as np


def predict(x, theta):
    """Predicts the output (y) given an matrix of x parameters.
    The hypothesis function for the model.
    :param x:     the feature matrix or test data
    :param theta: the constant weights for the function
    :return the predicted values computed by the function (theta' * x)
    """
    return np.dot(x, theta)
