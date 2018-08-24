import numpy as np
from linear_regression import LinearRegression
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
        Initializes the linear model and sets polynomial features out of it
        :param linear_model: linear model to base the polynomial regression off of
        :param degree: the highest degree polynomial feature in the model
        """
        self.linear_model = LinearRegression(linear_model.x, linear_model.y)
        self.poly_features = self.generate_polynomial(degree)
        self.upper_degree = degree
        self.linear_model.x = self.generate_polynomial(self.upper_degree)

    def generate_polynomial(self, degree):
        """
        Generates polynomial features for each column in the training matrix up to the degree specified
        :param degree: the highest degree polynomial feature
        :return: a matrix with polynomial features
        """
        m = np.size(self.linear_model.x, axis=0)
        width = np.size(self.linear_model.x[:, 1:], axis=1)
        poly_mat = np.ones((m, width * degree))
        # For loop could be vectorized
        for d in range(1, degree + 1):
            poly_mat[:, degree - 1 * width:degree * width] = (self.linear_model.x[:, 2:] ** degree)
        poly_mat = np.concatenate([np.ones((np.size(poly_mat, axis=0), 1)), poly_mat], axis=1)

        return poly_mat

    def predict(self, x, theta):
        """
        This function transforms linear features into polynomial features and calculates the predicted values from
        the regression model using the given theta values
        :param x: the linear feature matrix (Dimensions: m x n)
        :param theta: the theta vector      (Dimensions: t x 1)
        :return: the values predicted by the model for each row of features (Dimensions: t x 1)
        """
        # First generate polynomial values from the x linear features
        m = np.size(x, axis=0)
        width = np.size(x[:, 1:], axis=1)
        x_poly = np.ones((m, width * self.upper_degree))
        for d in range(1, self.upper_degree + 1):
            x_poly[:, self.upper_degree - 1 * width:self.upper_degree * width] = (x[:, 1:] ** self.upper_degree)

        return self.linear_model.predict(x_poly, theta)

if __name__ == "__main__":
    data = np.genfromtxt('train_linear.csv', delimiter=',')

    linear = LinearRegression(data[1:, 0:1], data[1:, 1:2])
    model = PolynomialRegression(linear, degree=3)
    theta_size = np.size(model.linear_model.x, axis=1)

    model.theta, cost, grads = model.linear_model.gradient_descent(.0005, np.zeros((theta_size, 1)),
                                                                   model.linear_model.x, model.linear_model.y,
                                                                   model.linear_model.cost)
    print(model.theta, cost, grads)