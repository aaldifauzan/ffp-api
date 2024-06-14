import numpy as np
from numpy.linalg import pinv

class elm():
    '''
    A class for Extreme Learning Machine for regression problems.
    '''
    def __init__(self, hidden_units, activation_function, x, y, random_type):
        self.hidden_units = hidden_units
        self.activation_function = activation_function
        self.x = x
        self.y = y
        self.beta = np.zeros((self.hidden_units, 1))
        self.random_type = random_type

        # Randomly generate the weight matrix and bias vector from input to hidden layer
        if self.random_type == 'uniform':
            self.W = np.random.uniform(low=0, high=1, size=(self.hidden_units, self.x.shape[1]))
            self.b = np.random.uniform(low=0, high=1, size=(self.hidden_units, 1))
        if self.random_type == 'normal':
            self.W = np.random.normal(loc=0, scale=0.5, size=(self.hidden_units, self.x.shape[1]))
            self.b = np.random.normal(loc=0, scale=0.5, size=(self.hidden_units, 1))

    def __input2hidden(self, x):
        '''
        Compute the output of hidden layer using different activation functions.
        '''
        temH = np.dot(self.W, x.T) + self.b
        if self.activation_function == 'sigmoid':
            return 1 / (1 + np.exp(-temH))
        elif self.activation_function == 'relu':
            return np.maximum(0, temH)
        elif self.activation_function == 'tanh':
            return np.tanh(temH)
        elif self.activation_function == 'leaky_relu':
            return np.maximum(0, temH) + 0.01 * np.minimum(0, temH)

    def fit(self):
        '''
        Train the model using a regularization technique.
        '''
        H = self.__input2hidden(self.x)
        # Regularization with identity matrix scaled by the inverse of C
        self.beta = np.dot(pinv(np.dot(H, H.T)), np.dot(H, self.y))

    def predict(self, x):
        H = self.__input2hidden(x)
        return np.dot(H.T, self.beta)

    def score(self, x_test, y_test):
        y_pred = self.predict(x_test)
        mae = np.mean(np.abs(y_pred - y_test))
        mse = np.mean((y_pred - y_test) ** 2)
        r_squared = 1 - (np.sum((y_test - y_pred) ** 2) / np.sum((y_test - np.mean(y_test)) ** 2))
        return mae,mse,r_squared