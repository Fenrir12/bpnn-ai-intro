import numpy as np
import matplotlib.pyplot as plt
import src.numtofeature as ntf


class MLP:
    def __init__(self, n_input, n_hidden, n_output, n_layer):
        '''Initializes weight matrices with respect to specified number of neurons
        for input, hidden and output layer. Also specifies the number of hidden layer
        of the network. Provides accessibility to the insides of the network with
        object variables of weights (W), activations(A), errors(E)
        and partial derivatives of errors (D)
        n_input : number of input neurons
        n_hidden : number of neurons in a hidden layer
        n_output : number of output neurons
        n_layer : number of layers including hidden and output layers
                (i.e. 4 hidden => n_layer = 4)'''

        # Set parameters of network
        self.n_input = n_input
        self.n_hidden = n_hidden
        self.n_output = n_output
        self.n_layer = n_layer + 2
        self.W = [0.0] * self.n_layer
        self.A = [0.0] * self.n_layer
        self.E = [0.0] * self.n_layer
        self.D = [0.0] * self.n_layer
        self.B = [0.0] * self.n_layer

        # Initialize weights and biases of layers according to depth of network
        if self.n_layer < 1:
            raise ValueError('n_layer should be greater or equal to 2')
        if self.n_layer == 2:
            self.W[0] = np.random.random((self.n_input, self.n_output))
            self.B[0] = np.zeros((1, self.n_output))
        else:
            self.W[0] = np.random.random((self.n_input, self.n_hidden))
            self.B[0] = np.zeros((1, self.n_hidden))
            self.W[self.n_layer - 1] = np.random.random((self.n_hidden, self.n_output))
            self.B[self.n_layer - 1] = np.zeros((1, self.n_hidden))
            for depth in range(1, self.n_layer - 1):
                self.W[depth] = np.random.random((self.n_hidden, self.n_hidden))
                self.B[depth] = np.zeros((1, self.n_hidden))

    def _sigmoid(self, x):
        '''Computes sigmoid of a value'''
        return 1 / (1 + np.exp(-x))

    def _dsigmoid(self, y):
        '''Computes derivative of sigmoid function'''
        return y * (1.0 - y)

    def _softmax(self, x):
        """Compute softmax values for each sets of scores in x."""
        return np.exp(x) / np.sum(np.exp(x), axis=0)

    def feedforward(self, inputs):
        '''Fills the activation matrix of the neurons
        and outputs the results at the end for one epoch'''
        for depth in range(self.n_layer - 1):
            # Compute activation of first hidden layer
            if depth == 0:
                self.A[depth] = self._sigmoid(np.dot(inputs, self.W[depth]))
            # Activation of every other layer
            else:
                self.A[depth] = self._sigmoid(np.dot(self.A[depth - 1], self.W[depth]))
        self.A[-1] = self._sigmoid(np.dot(self.A[-2], self.W[-1]))

        return self.A[-1]

    def fit(self, inputs, targets, learning_rate=0.01, n_epochs=200000):
        '''Train the weights of a custom network by computing activations from feedforward
        and then backpropagating the errors after one epoch. Uses simple error as loss function.'''
        ERROR = []

        for j in range(n_epochs):
            # Apply feed forward for epoch
            for depth in reversed(range(self.n_layer)):
                for idx, batch in enumerate(inputs):
                    self.feedforward(batch)
                    # BACKPROPAGATION
                    # Squared euclidean distance cost function
                    # Compute the error and derivative error of output layers' neurons
                    if depth == self.n_layer - 1:
                        self.E[depth] = targets[idx] - self.A[depth] # Derivative of the squared euclidean distance
                        self.D[depth] = np.multiply(self.E[depth], self._dsigmoid(self.A[depth]))
                    # Compute the error and derivative error of hidden layers' neurons
                    else:
                        self.E[depth] = np.dot(self.D[depth + 1], self.W[depth + 1].T)
                        self.D[depth] = self.E[depth] * self._dsigmoid(self.A[depth])

                    # Update weights based on contribution of neuron to error
                    if depth == 0:
                        self.W[depth] += np.dot(batch.T, self.D[depth]) * learning_rate
                    else:
                        self.W[depth] += np.dot(self.A[depth - 1].T, self.D[depth]) * learning_rate

                    if (j % 1000) == 0:
                        print("Error:" + str(np.mean(np.abs(self.E[self.n_layer - 1]))))
                        ERROR.append(np.mean(np.abs(self.E[-1])))

                        plt.xlabel('Epochs (n)')
                        plt.ylabel('Cost function')
                        plt.plot(range(0, len(ERROR)), ERROR)
                        plt.pause(0.000001)
        plt.show()

    def predict(self, i):
        '''Returns the output of the network given an input matching input neurons'''
        return self.feedforward(i)


if __name__ == "__main__":
    mlp = MLP(2, 3, 4, 1)
    # inputs = np.array([[[0, 0], [0, 1], [1, 0], [1, 1]],
    #                    [[0, 0], [0, 1], [1, 0], [1, 1]],
    #                    [[0, 0], [0, 1], [1, 0], [1, 1]],
    #                    [[0, 0], [0, 1], [1, 0], [1, 1]]])
    # targets = np.array([[[0, 0, 0, 0], [1, 0, 0, 0], [1, 0, 0, 0], [0, 0, 0, 0]],
    #                     [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 1, 0, 0]],
    #                     [[0, 0, 0, 0], [0, 0, 1, 0], [0, 0, 1, 0], [0, 0, 1, 0]],
    #                     [[0, 0, 0, 1], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]]])

    answer_key, dgtLst = ntf.read_digit_file('data/smalltrain.csv', 2, 2)
    mlp.fit(inputs, targets, learning_rate=0.1)
    print(str(mlp.predict([0, 0])))
    print(str(mlp.predict([0, 1])))
    print(str(mlp.predict([1, 0])))
    print(str(mlp.predict([1, 1])))
