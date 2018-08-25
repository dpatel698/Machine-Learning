import numpy as np
from math import exp, log
from general_learning import predict
import matplotlib.pyplot as plt


def sigmoid(z):
    '''
    The sigmoid function maps elements of z to discrete values in the [0,1] range. This property allows the classifier
    to treat the data points as probabilities and set a threshold where a data point is classified as y = 0 or y = 1.

    :param z: vector of data points to apply the sigmoid function to
    :return: the vector containing the sigmoid
    '''
    return np.vectorize(lambda x: 1 / (1 + exp(-x)))(z)


def logistic_cost(x, y, theta):
    """
    The logistic cost function uses cross entropy which combines two logarithmic for cases where y = 0 and y = 1
    into a single convex cost function. This gives the error in the model and a function that can be
    optimized efficiently.

    :param x: the feature matrix (Dimensions: m x n)
    :param y: the label (target) data (Dimensions: m x 1)
    :param theta: the constant weights for the hypothesis of the model (Dimensions: n x 1)
    :return: the cost of the hypothesis for the feature and label data
    """
    observations = np.size(x, axis=1)
    hypothesis = predict(x, theta)  # m x 1
    y_1 = np.multiply(y, np.vectorize(log)(hypothesis))
    y_0 = np.multiply(np.subtract(1, y), np.vectorize(log)(1 - hypothesis))
    return ((-1 / observations) * np.sum(y_0 + y_1)).item(0)


def logistic_cost_regularized(x, y, theta, reg_param):
    """
    A regularized version of the logistic cost function to help prevent overfitting

    :param x: the feature matrix (Dimensions: m x n)
    :param y: the label (target) data (Dimensions: m x 1)
    :param theta: the constant weights for the hypothesis of the model (Dimensions: n x 1)
    :param reg_param: the regularization parameter
    :return: the cost of the hypothesis for the feature and label data
    """
    observations = np.size(x, axis=1)
    hypothesis = predict(x, theta)  # m x 1
    y_1 = np.multiply(y, np.vectorize(log)(hypothesis))
    y_0 = np.multiply(np.subtract(1, y), np.vectorize(log)(1 - hypothesis))
    regularization = (reg_param / (2 * observations)) * np.sum(np.power(theta[1:], 2))
    return ((-1 / observations) * np.sum(y_0 + y_1) + regularization).item(0)


def plot_decision_boundary(x1, x2, y, theta, x1_label='x1', x2_label='x2', plot_title='No Title'):
    """
    Plots all the data points (x1,x2) and draws a decision boundary that attempts to separate (x1,x2) where y = 0
    and y == 1
    :param x1: values for the x-axis (Dimensions: m x 1)
    :param x2: values for the y-axis (Dimensions: m x 1)
    :param y:  the classification for the rows in x1/x2 (y = 1 or y = 0)  (Dimensions: m x 1)
    :param theta: the constants for the logistic classifier (Dimensions: 2 x 1)
    :param x1_label: title for the x-axis
    :param x2_label: title for the y-axis
    :param plot_title: title for the entire plot
    """
    fig = plt.figure()
    axis = fig.add_subplot(111)

    x = np.concatenate([x1, x2], axis=1)
    positives = x[np.where(y == 1)[0], :]
    negatives = x[np.where(y == 0)[0], :] - 1

    # Plot the data points as a scatter plot where positive/negative results are differentiated by shape and color
    axis.scatter(positives[:, 0], positives[:, 1], marker='+', c='b')
    axis.scatter(negatives[:, 0], negatives[:, 1], marker='_', c='r')

    # Set the x and y axis titles
    axis.set_xlabel(x1_label)
    axis.set_ylabel(x2_label)

    # Set the graph's title and legend location
    axis.set_title(plot_title)
    plt.legend(loc='upper right')

    plt.show()

plot_decision_boundary(np.random.rand(500, 1), np.random.rand(500, 1), np.random.randint(2, size=(500, 1)), None,
                       plot_title='Logistic Classifier')

