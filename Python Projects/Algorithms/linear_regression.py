import numpy as np
from general_learning import predict
from gradient_descent import batch_gradient_descent
import matplotlib.pyplot as plt


class LinearRegression:
    """
    This class models a linear function to a data set and contains functions to optimize the model, predict
    values from the model, and input data to train and further improve the model
    """
    # Our feature matrix x will contain the training examples used to fit the hypothesis for the linear model
    # (Dimensions: m x n)
    x = None

    # The target examples (Dimensions: m x 1)
    y = None

    # The constant parameters for the model's hypothesis (Dimensions: n x 1)
    theta = None

    def __init__(self, x=None, y=None):
        """
        The constructor initializes the feature and target examples.

        :param x: the feature matrix
        :param y: the target example matrix
        """
        self.x = np.concatenate([np.ones((np.size(x, axis=0), 1)), x], axis=1)
        self.y = y
        self.__feature_normalize()

    def __feature_normalize(self):
        """
        Scales values of features in the training set proportionally equal values so any relatively large
        feature values do not dominate the optimization of the linear function.
        """
        # First we find the mean and standard deviation for each column of the feature set
        mean = np.mean(self.x[:, 1:], axis=0)  # 1 x n
        std = np.std(self.x[:, 1:], axis=0)  # 1 x n
        self.x[:, 1:] = np.divide(self.x[:, 1:] - mean, std)

    def cost(self, x, y, theta):
        """
        This function evaluates the cost of the linear function which is the sum of the squared errors of all
        the training examples predicted by the hypothesis function and the target data
        :param x: the test data   (Dimensions: m x n)
        :param y: the target data (Dimensions: m x 1)
        """
        observations = np.size(x, axis=0)
        squared_error = (predict(x, theta) - y) ** 2
        squared_error.sum(axis=0)

        return squared_error.sum(axis=0) / observations

    def simple_regression(self, x, y, theta):
        """
        Plot the line of best fit predicted by the model on the given data
        :param x: the features used to test the model       (Dimensions: m x 1)
        :param y: the target data for the model             (Dimensions: m x 1)
        :param theta: the weights the model predicts with   (Dimensions: 2 x 1)
        """
        # Set up the figure to visualize the data and model
        fig = plt.figure()
        ax = fig.add_subplot(111)

        # Plot the data points (x, y) for the given data as a scatterplot
        ax.scatter(x[:, 1], y, marker='o', c='r')

        observations = np.size(x, axis=0)  # m data points

        # Create the line of best fit using the theta values and predicting over an interval
        x_fit = np.linspace(x[:, 1].min(), x[:, 1].max() + 10, 50)
        y_fit = theta[0] + (x_fit * theta[1])
        plt.plot(x_fit, y_fit)

        plt.show()


if __name__ == '__main__':
    data = np.genfromtxt('train_linear.csv', delimiter=',')

    model = LinearRegression(data[1:, 0:1], data[1:, 1:2])

    model.theta, cost, grads = batch_gradient_descent(.0005, np.zeros((2, 1)), model.x, model.y, model.cost)

    print(model.theta, cost, grads)

    model.simple_regression(model.x, model.y, model.theta)

