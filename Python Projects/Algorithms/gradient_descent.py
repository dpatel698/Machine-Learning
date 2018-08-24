import numpy as np


def batch_gradient_descent(learning_rate, theta, x, y, cost_func):
    """
    Performs a batch gradient descent algorithm on the training and target data in order to optimize the theta
    parameters for the linear function. Conceptually the gradient descent algorithm steps through the graph of
    theta values that minimize the cost function (J, least squared error) and in the optimal case converges on
    a global minimum.
    :param learning_rate: determines how much the gradient is scaled at each step in the algorithm
    :param theta:        the constant parameters of the function fitting the model (Dimensions: n x 1)
    :param x:            the feature values (Dimensions: m x n)
    :param y:            the target value   (Dimensions: m X 1)
    :param cost_func:    function the calculates the cost for this model
    :return: the optimized theta, new cost for the model, and steps taken in gradient descent
    """
    m = np.size(x, axis=0)
    prev_cost = cost_func(x, theta)
    next_cost = 0
    steps = 0
    while abs(prev_cost - next_cost) >= .00002:
        hypothesis = np.dot(x, theta)  # m x 1
        gradient = np.dot(np.transpose(x), (hypothesis - y))  # n x 1
        theta[0] -= learning_rate * (1 / m) * np.sum(np.dot(x, theta) - y, axis=0)
        theta[1:] -= (learning_rate * (1 / m) * gradient)[1:]
        prev_cost = next_cost
        next_cost = cost_func(x, theta)
        steps += 1

    return theta, next_cost, steps