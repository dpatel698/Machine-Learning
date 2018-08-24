import numpy as np


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
        The constructor initializes the feature and target examples
        :param x: the feature matrix
        :param y: the target example matrix
        """
        self.x = np.concatenate([np.ones((np.size(x, axis=0), 1)), x], axis=1)
        self.y = y
        self.__feature_normalize()

    @staticmethod
    def predict(x, theta):
        """Predicts the output (y) given an matrix of x parameters.
        The hypothesis function for the model.
        """
        return np.dot(x, theta)

    def __feature_normalize(self):
        """
        Scales values of features in the training set proportionally equal values so any relatively large
        feature values do not dominate the optimization of the linear function
        """
        # First we find the mean and standard deviation for each column of the feature set
        mean = np.mean(self.x[:, 1:], axis=0)  # 1 x n
        std = np.std(self.x[:, 1:], axis=0)  # 1 x n
        self.x[:, 1:] = np.divide(self.x[:, 1:] - mean, std)

    def cost(self, x, theta):
        """
        This function evaluates the cost of the linear function which is the sum of the squared errors of all
        the training examples predicted by the hypothesis function and the target data
        """
        m = np.size(x, axis=0)
        return np.sum((self.predict(x, theta) - self.y) ** 2, axis=0).flatten()[0] / m

    @staticmethod
    def gradient_descent(learning_rate, theta, x, y, cost_func):
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

if __name__ == "__main__":
    data = np.genfromtxt('train_linear.csv', delimiter=',')

    model = LinearRegression(data[1:, 0:1], data[1:, 1:2])

    model.theta, cost, grads = model.gradient_descent(.0005, np.zeros((2, 1)), model.x, model.y, model.cost)

    print(model.theta, cost, grads)