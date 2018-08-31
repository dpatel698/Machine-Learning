import numpy as np
import matplotlib.pyplot as plt
from linear_regression import LinearRegression
from general_learning import predict
from gradient_descent import batch_gradient_descent


# from sklearn.preprocessing import PolynomialFeatures


class PolynomialRegression:
    """
    This class performs polynomial regression to better fit a data set using non-linear features derived from a linear
    model.
    """

    # The linear model whose features we want to fit more efficiently
    linear_model = None

    # The highest degree for the polynomial regression to optimize (low/high degree may lead to greater bias/variance)
    upper_degree = 0

    # The constant parameters for the model's hypothesis (Dimensions: n x 1)
    theta = None

    def __init__(self, linear_model, degree=0):
        """
        Initializes the linear model and sets polynomial features out of it.

        :param linear_model: linear model to base the polynomial regression off of
        :param degree: the highest degree polynomial feature in the model
        """
        self.linear_model = linear_model
        self.upper_degree = degree
        self.poly_features = self.generate_polynomial(self.linear_model.x[:, 1:], degree)
        self.linear_model.x = self.poly_features

    def generate_polynomial(self, x, degree):
        """
        Generates polynomial features for each column in the training matrix up to the degree specified.

        :param      x: the features to make polynomial values from (should not include constant variable row, theta_0)
        :param degree: the highest degree polynomial feature
        :return: a matrix with polynomial features
        """
        m = np.size(x, axis=0)
        width = np.size(x, axis=1)
        poly_mat = np.ones((m, width * degree))
        # For loop could be vectorized
        for d in range(0, degree):
            poly_mat[:, d * width:(d + 1) * width] = (x ** (d + 1))

        return np.concatenate([np.ones((m, 1)), poly_mat], axis=1)

    def predict(self, x, theta):
        """
        This function transforms linear features into polynomial features and calculates the predicted values from
        the regression model using the given theta values.

        :param x:     the linear feature matrix  (Dimensions: m x n)
        :param theta: the constant weight vector (Dimensions: d x 1)
        :return: the values predicted by the model for each row of features (Dimensions: d x 1)
        """
        # First generate polynomial values from the x linear features
        m = np.size(x, axis=0)
        width = np.size(x[:, 1:], axis=1)
        x_poly = np.ones((m, width * self.upper_degree))
        for d in range(0, self.upper_degree):
            x_poly[:, d * width:(self.upper_degree + 1) * width] = (x ** (d + 1))

        return predict(x_poly, theta)

    def simple_polynomial(self, x, y, theta):
        """
        Plot the line of best fit predicted by the model on the given data
        :param x: the features used to test the model       (Dimensions: m x n)
        :param y: the target data for the model             (Dimensions: m x 1)
        :param theta: the weights the model predicts with   (Dimensions: n x 1)
        """
        # Set up the figure to visualize the data and model
        fig = plt.figure()
        ax = fig.add_subplot(111)

        # Plot the data points (x, y) for the given data as a scatterplot
        ax.scatter(x, y, marker='o', c='r')

        # Create the line of best fit using the theta values and predicting over an interval
        x_deg = np.linspace(x.min(), x.max(), 100)
        x_deg = np.reshape(x_deg, (x_deg.size, 1))
        x_fit = self.generate_polynomial(x_deg, self.upper_degree)
        y_fit = predict(x_fit, theta)
        plt.plot(x_deg, y_fit)

        plt.show()


if __name__ == '__main__':
    data = np.genfromtxt('train_linear.csv', delimiter=',')

    linear = LinearRegression(data[:, 0:1], data[:, 1:2])

    model = PolynomialRegression(linear, degree=4)
    theta_size = np.size(model.linear_model.x, axis=1)
    model.theta, cost, grads = batch_gradient_descent(.1**3, np.zeros((theta_size, 1)),
                                                      model.linear_model.x, model.linear_model.y,
                                                      model.linear_model.cost)
    print(model.theta, cost, grads)

    model.simple_polynomial(data[:, 0:1], model.linear_model.y, model.theta)