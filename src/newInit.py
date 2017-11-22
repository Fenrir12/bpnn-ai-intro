   #lyrNodes should be list of int's of number of nodes in each layer ordered from 
    #input to out so [input, hidden1, hidden2..., output]
     def __init__(self, lyrNodes, lrnRate, regParam):
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
        self.n_input = lyrNodes[0]
        self.n_output = lyrNodes[-1]
        self.n_layer = len(lyrNodes)-1
        self.alpha = lrnRate         #learning rate paarameter
        self.lmda = regParam         #regularization parameter
        self.cost = 0
        self.W = [0.0] * (self.n_layer) # Weight of neurons in each layers
        self.A = [0.0] * self.n_layer # Output of neurons in each layers
        self.dC = [0.0] * (self.n_layer) # Derivative of cost function
        self.dW = [0.0] * (self.n_layer) # Rate of change of cost function for each neurons
        self.B = [0.0] * self.n_layer # Bias term of each neuron

        
        
        # Initialize weights and biases of layers according to depth of network
        if self.n_layer < 1:
            raise ValueError('n_layer should be greater or equal to 1')
        
            
        if self.n_layer == 1:
            self.W[0] = np.random.normal(scale=0.1, size=(self.n_input, self.n_output))
            self.B[0] = np.random.normal(scale=0.1, size=(1, self.n_output))
        else:
            for depth in range(0, len(lyrNodes)-1):
                self.W[depth] = np.random.normal(scale=0.1, size=(lyrNodes[depth], lyrNodes[depth+1]))
                self.B[depth] = np.random.normal(scale=0.1, size=(1, lyrNodes[depth+1]))