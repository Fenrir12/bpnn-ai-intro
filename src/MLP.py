import numpy as np


class MLP:
    def __init__(self, n_input, n_hidden, n_output, n_layer):
        self.n_input = n_input
        self.n_hidden = n_hidden
        self.n_output = n_output
        self.n_layer = n_layer + 1
        self.W = [0.0] * self.n_layer
        self.A = [0.0] * self.n_layer
        self.E = [0.0] * self.n_layer
        self.deltas = [0.0] * self.n_layer

        # Initialize weights of layers according to depth of network
        if self.n_layer < 1:
            raise ValueError('n_layer should be greater or equal to 2')
        if self.n_layer == 1:
            self.W[0] = np.random.random((self.n_input, self.n_output))
        else:
            self.W[0] = np.random.random((self.n_input, self.n_hidden))
            self.W[self.n_layer - 1] = np.random.random((self.n_hidden, self.n_output))
            for depth in range(1, self.n_layer - 2):
                self.W[depth] = np.random.random((self.n_hidden, self.n_hidden))


    # Computes sigmoid of a value
    def _sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    # Computes derivative of sigmoid function
    def _dsigmoid(self, y):
        return y * (1.0 - y)

    def feedforward(self, inputs):
        # FEED-FORWARD
        for depth in range(self.n_layer):
            # Compute activation of first hidden layer
            if depth == 0:
                self.A[depth] = self._sigmoid(np.dot(inputs, self.W[depth]))
            else:
                self.A[depth] = self._sigmoid(np.dot(self.A[depth - 1], self.W[depth]))
        return self.A[-1]
    # Train the weights of the neural network
    def fit(self, inputs, targets, learning_rate=1.00, n_epochs=200000):
        for j in range(n_epochs):
            # Apply feed forward for epoch
            self.feedforward(inputs)

            # BACKPROPAGATION
            # Squared euclidean distance loss function
            for depth in reversed(range(self.n_layer)):

                # Compute the contribution of neuron to error of an latter neuron
                if depth == self.n_layer - 1:
                    self.E[depth] = targets - self.A[depth]
                    self.deltas[depth] = self.E[depth] * self._dsigmoid(self.A[depth])
                else:
                    self.E[depth] = np.dot(self.deltas[depth + 1], self.W[depth + 1].T)
                    self.deltas[depth] = self.E[depth] * self._dsigmoid(self.A[depth])

                # Update weights based on contribution of neuron to error
                if depth == 0:
                    self.W[depth] += np.dot(inputs.T, self.deltas[depth]) * learning_rate
                else:
                    self.W[depth] += np.dot(self.A[depth - 1].T, self.deltas[depth]) * learning_rate
            if (j % 10000) == 0:
                print("Error:" + str(np.mean(np.abs(self.E[self.n_layer-1]))))
    def predict(self, i):
        return self.feedforward(i)

if __name__ == "__main__":
    mlp = MLP(2, 3, 1, 1)
    inputs = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
    targets = np.array([[0], [1], [1], [0]])
    mlp.fit(inputs, targets)
    print(str(mlp.predict([0, 1])))