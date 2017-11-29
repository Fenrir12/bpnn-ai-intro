import numpy as np
import matplotlib.pyplot as plt
# import src.numtofeature as ntf
import numtofeature as ntf
import pandas as pd
from random import sample

def wait():
    # programPause = raw_input("Press the <ENTER> key to continue...")
    programPause = input("Press the <ENTER> key to continue...")


class MLP:
    def __init__(self, lyrNodes, lrnRate=0.1, batch_size=10,  n_epochs=200, 
                  hiddenActv='sigmoid', outputActv='softmax', costFnc='sqrdEuc', 
                   regMethod ='dropout', regParam=0.2, dropPrtn=0.2, momentum=1):
        
        '''Initializes weight matrices with respect to specified number of neurons
        for input, hidden and output layer. Also specifies the number of hidden layer
        of the network. Provides accessibility to the insides of the network with
        object variables of weights (W), activations(A), errors(E)
        and partial derivatives of errors (D)
        lyrNodes: list of ints specifying number of Nodes(neurons), in each layer
                    order of [layer input, hidden0,hidden1...hiddenN,Output]
        lrnRate = learning rate, alpha, for weight updates in backprops
        batch_size = number of training examples into each batch
        
        n_epochs = number of training iterations to use with backprop
        costFunc : 'xentropy' or 'sqrdEuc'    
        outputActv : 'softmax' or 'softplus' or 'sigmoid'
        hiddenActv : 'softmax' or 'softplus' or 'sigmoid'
        regMethd : 'weightedDst' or 'dropout'
        regParam : regularization parameter used for weighted distance, if using dropout
                    set to zero can be any int, but higher number may degrade accuracy
        dropPrtn : portion of neurons in hidden layer to dropout for regularization. 
                    Not used in weightedDist, must be a decimal recomended .1=.5 
        momentum : momentum factor to change learning rate in back prop
                    based on magnitude of last error.  If don't want to use set
                    to zero. Any int.
        '''

        # Set parameters of network
        self.n_input = lyrNodes[0]
        self.n_output = lyrNodes[-1]
        self.n_layer = len(lyrNodes) - 1
        self.batch_size = batch_size # Number of samples going through one feedforward and backprop
        self.hiddenActv = hiddenActv # Activation function for hidden layers
        self.outputActv = outputActv # Activation function for output layer
        self.costFnc = costFnc # Cost function used for error and gradient descent
        self.n_epochs = n_epochs # Number of epochs for learning
        self.alpha = lrnRate  # learning rate parameter
        
        
        self.mntmRate = momentum # Momentum factor
        self.cost = 0 # initialize cost value
        self.W = [0.0] * self.n_layer  # Weight of neurons in each layers
        self.A = [0.0] * self.n_layer  # Output of neurons in each layers
        self.dC = [0.0] * self.n_layer  # Derivative of cost function
        self.dW = [0.0] * self.n_layer  # Rate of change of cost function for each neurons
        self.update = [0.0] * self.n_layer # Store update vector for next update with momentum
        self.B = [0.0] * self.n_layer  # Bias term of each neuron
        self.regMthd = regMethod    #'dropout' or 'weightDist'
        self.drpLst = []   #lst of neurons to set to dropout from FF and Backprop
        self.drpPrtn = dropPrtn  #what portion of hidden neurons in each hidden layer to turn off
        self.lmda = regParam  # regularization parameter
        
        # Initialize weights and biases of layers according to depth of network
        if self.n_layer < 1:
            raise ValueError('n_layer should be greater or equal to 1')
        
        #if there is no hidden layer
        if self.n_layer == 1:
            self.W[0] = np.random.normal(scale=0.1, size=(self.n_input, self.n_output))
            self.B[0] = np.random.normal(scale=0.1, size=(1, self.n_output))
            self.update[0] = np.zeros((self.n_input, self.n_output))   
        else:
            for depth in range(0, len(lyrNodes) - 1):
                self.W[depth] = np.random.normal(scale=0.1, size=(lyrNodes[depth], lyrNodes[depth + 1]))
                self.B[depth] = np.random.normal(scale=0.1, size=(1, lyrNodes[depth + 1]))
                self.update[depth] = np.zeros((1, lyrNodes[depth + 1]))

    #########   Explanation  #######################
    #   This function will test a neural network against test data
    #########   Input  #######################
    # trnLst: list of 1D numpy vectors, each vec holds features of 1 digit image
    # trnKey: list of 1d numpy vectors, each vec holds binary representation of 
    #           digits 0-9, position of 1 is the digit.  Corresponds by position with
    #########   Output  #######################
    # Will print out overall accuracy rate, accuracy rate of each digit
    def testAccuracy(self, tstLst, tstKey):
        i = 0
        inaccurate = 0

        for i in range(len(tstLst)):
            predict = self.feedforward(tstLst[i])
            maxpos = np.argmax(predict)
            pretty = np.zeros_like(predict)
            pretty[0][maxpos] = 1

            if (maxpos != np.argmax(tstKey[i])):
                inaccurate += 1.0
        accuracy = 100 - 100 * float(inaccurate / len(tstLst))
        return accuracy

    # jmw
    # this function takes a matrix, threshholds and scales it accrodingly
    def threshold(self, matrix, lowthresh=.001, upthresh=1):
        mtrxmin = np.amin(matrix)
        if mtrxmin < lowthresh:
            minmult = lowthresh / mtrxmin
            matrix = matrix * minmult
        mtrxmax = np.amax(matrix)
        if mtrxmax > upthresh:
            maxmult = upthresh / mtrxmax
            matrix = matrix * maxmult
        return matrix

    # jmw
    # elementwise version of ReLU equation
    def softplusElement(self, x):
        # approximate of ReLU that you can take a derivative of
        # https://www.quora.com/What-is-special-about-rectifier-neural-units-used-in-NN-learning
        tmp = 1 + np.exp(x)
        y = np.log(tmp)
        return y

    # jmw
    # elementwise version of derivative of ReLU equation
    def dsoftplusElement(self, x):
        y = 1 / (1 + np.exp(-x))
        return y

    # matix calc for activation function
    def _softplus(self, x):
        # approximate of ReLU that you can take a derivative of
        return np.log(1 + np.exp(x))

    # matrix calc used in backpropagation
    def _dsoftplus(self, x):
        return 1 / (1 + np.exp(-x))

    def _sigmoid(self, x):
        '''Computes sigmoid of a value'''
        return 1 / (1 + np.exp(-x))

    def _dsigmoid(self, y):
        '''Computes derivative of sigmoid function'''
        return y * (1.0 - y)

    def _softmax(self, batch_o):
        return np.array([np.exp(x) / np.exp(x).sum() for x in batch_o])

    def _activation(self, x, activationFct, derivative=False):
        if activationFct == 'sigmoid':
            if derivative:
                return self._dsigmoid(x)
            else:
                return self._sigmoid(x)
        elif activationFct == 'softplus':
            if derivative:
                return self._dsoftplus(x)
            else:
                return self._softplus(x)
        elif activationFct == 'softmax':
            if derivative:
                return self._dsigmoid(x)
            else:
                return self._softmax(x)

    def costfunc(self, targets, outputs, derivative=0):
        if self.costFnc == 'sqrdEuc':
            if derivative == 1:
                return -(targets - outputs)
            else:
                return 0.5 * (targets - outputs) ** 2
        elif self.costFnc == 'xentropy':
            if derivative == 1:
                return (outputs - targets)
            else:
                return targets * np.log(outputs)

    def feedforward(self, inputs):
        '''Fills the activation matrix of the neurons
        and outputs the results at the end for one epoch with chosen activation function'''
        self.dropMsk = [0]*self.n_layer
        for depth in range(self.n_layer):
            # Compute activation of first hidden layer
            if depth == 0:
                self.A[depth] = self._activation(np.dot(inputs, self.W[depth]) + self.B[depth], self.hiddenActv)
                if (self.regMthd=='dropout'):
                    #randomly select positions which will be turned off
                    numNrns = len(self.A[depth][0])
                    trnoff = sample(range(0,numNrns), int(round(self.drpPrtn*numNrns, 0)))
                    for x in trnoff:
                        for row in self.A[depth]:
                            row[x]=0

            # Activation of every other layer
            else:
                self.A[depth] = self._activation(np.dot(self.A[depth - 1], self.W[depth]) + self.B[depth],
                                                 self.hiddenActv)
                if (self.regMthd=='dropout'):
                    #randomly select positions which will be turned off
                    numNrns = len(self.A[depth][0])
                    trnoff = sample(range(0,numNrns), int(round(self.drpPrtn*numNrns, 0)))
                    for x in trnoff:
                        for row in self.A[depth]:
                            row[x]=0
                
        #handling output layer
        self.A[depth] = self._activation(np.dot(self.A[depth - 1], self.W[depth]) + self.B[depth], self.outputActv)
        # return output layer
        return self.A[depth]

    def backprop(self, inputs, targets_b):
        for depth in reversed(range(self.n_layer)):
            # Compute the error and derivative error of output layers' neurons
            if depth == self.n_layer - 1:
                self.cost = self.costfunc(targets_b, self.A[depth])
                self.dC[depth] = self.costfunc(targets_b, self.A[depth],
                                               derivative=True)  # Derivative of the squared euclidean distance
                if self.costFnc == 'sqrdEuc':
                    self.cost = np.sum(self.cost)
                    self.dW[depth] = np.multiply(self.dC[depth], self._activation(self.A[depth], self.outputActv,
                                                                                      derivative=True))  # Haddamard product
                elif self.costFnc == 'xentropy':
                    self.cost = -np.mean(self.cost)
                    self.dW[depth] = self.dC[depth]
                    
            # Compute the error and derivative error of hidden layers' neurons
            else:
                self.dC[depth] = np.dot(self.dW[depth + 1], self.W[depth + 1].T)
                # adjust dC to threshhold values
                self.dC[depth] = self.threshold(self.dC[depth])

                self.dW[depth] = np.multiply(self.dC[depth], self._activation(self.A[depth], self.hiddenActv,
                                                                              derivative=True))  # Haddamard product

            # Update weights based on contribution of neuron to error and weight regularization
            # if input layer
            if (depth == 0):
                self.update[depth] = (np.dot(inputs.T, self.dW[depth])) * self.alpha + self.mntmRate*self.update[depth]
                
                regularized = self.W[depth] * (1 - self.alpha * self.lmda / len(inputs))
                self.W[depth] = regularized - self.update[depth]

            if (depth != 0):
                self.update[depth] = (np.dot(self.A[depth - 1].T, self.dW[depth])) * self.alpha + self.mntmRate*self.update[depth]
                
                regularized = self.W[depth] * (1 - self.alpha * self.lmda / len(inputs))
                self.W[depth] = regularized - self.update[depth]

            # then update bias
            self.B[depth] -= np.mean(self.dW[depth], axis=0) * self.alpha

    def fit(self, inputs, targets, tstData, tstKey):
        '''Train the weights of a custom network by computing activations from feedforward
        and then backpropagating the errors after one epoch. Uses simple error as loss function.'''
        ERROR = []
        trnErr = []
        tstErr = []
        batch_size = self.batch_size
        numsteps = int(len(inputs) / batch_size) - 1
        print("num inputs", len(inputs))
        print("numsteps", numsteps)

        for j in range(self.n_epochs):
            btchstp = j % numsteps
            targets_b, batch = ntf.batchify(inputs, targets, batch_size, btchstp)
            # Apply feed forward for epoch
            self.feedforward(batch)

            # Update weights and bias with backpropagation
            self.backprop(batch, targets_b)

            ERROR.append(self.cost)

            trnErr.append(self.testAccuracy(inputs, targets))
            tstErr.append(self.testAccuracy(tstData, tstKey))
            
            '''
            # print("Trn Error,", trnErr,"Tst Error,", tstErr)
            if (j % 5 == 0):
                # printlst = []
                # printlst.append(ERROR)
                # printlst.append(trnErr)
                # printlst.append(tstErr)
                # printlst = np.array(printlst)
                # printDF = pd.DataFrame(printlst)
                # csvname = "ErrRcrdFtrs" + str(len(inputs[0])) + "Hdn" + str(self.n_hidden) + \
                #           "Trn" + str(len(targets)) + "Tst" + str(len(tstData)) + "It" + str(j) + ".csv"
                # printDF.to_csv(csvname)

                # Show cost function and performance results
                plt.figure(1)
                plt.subplot(211)
                plt.xlabel('Epochs (n)')
                plt.ylabel('Cost function')
                plt.plot(range(0, len(ERROR)), ERROR)
                plt.subplot(212)
                plt.xlabel('Epochs (n)')
                plt.ylabel('Performance')
                plt.plot(range(0, len(trnErr)), trnErr, 'bo')
                plt.plot(range(0, len(tstErr)), tstErr, 'r--')
                plt.pause(0.000001)
        '''
        return ERROR, trnErr, tstErr
        #plt.show()


if __name__ == "__main__":
    # n_input, n_hidden, n_output, n_layer (optional specify learning rate and reg param)
    prnttst = []
    prnttrn = []
    prnterr = []
    
    print ('==========Extracting train set')
    trnLst, trnKey, tstLst, tstKey = \
        ntf.read_trn_partial('data/mnsitTrain.csv', 28, 28, 40000, 2000)
            
    #testing lambda reg parameter
    for i in range(4):
        mlp = MLP([784, 135, 10])
        mlp.hiddenActv = 'softplus'
        mlp.batch_size = 17
        mlp.n_epochs = 800
        mlp.alpha = 0.01
        mlp.mntmRate = 0.0
        mlp.outputActv = 'softmax'
        mlp.costFnc = 'xentropy'
        mlp.regMthd = 'weightedDst'
        mlp.lmda = 0 + i
        # Test XOR, AND, OR and NOR inputs and targets
        # inputs = np.array([[[0, 0], [0, 1], [1, 0], [1, 1]],
        #                    [[0, 0], [0, 1], [1, 0], [1, 1]],
        #                    [[0, 0], [0, 1], [1, 0], [1, 1]],
        #                    [[0, 0], [0, 1], [1, 0], [1, 1]]])
        # targets = np.array([[[0, 0, 0, 0], [1, 0, 0, 0], [1, 0, 0, 0], [0, 0, 0, 0]],
        #                     [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 1, 0, 0]],
        #                     [[0, 0, 0, 0], [0, 0, 1, 0], [0, 0, 1, 0], [0, 0, 1, 0]],
        #                     [[0, 0, 0, 1], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]]])


        # print(trnLst[0].shape)
        print('==========Training Session: '+str(i))
        print('Number of hidden layers = ' + str(mlp.n_layer - 1))
        print('Hidden layers activation function = ' + str(mlp.hiddenActv))
        print('Output layer activation function = ' + str(mlp.outputActv))
        print('Regularization Method = ' + mlp.regMthd)
        print('Batch size = ' + str(mlp.batch_size))
        print('Cost function = ' + str(mlp.costFnc))
        print('Training epochs = ' + str(mlp.n_epochs))
        
        error, trnAcc, tstAcc = mlp.fit(trnLst, trnKey, tstLst, tstKey)
        
        prnttst.append(tstAcc)
        prnttrn.append(trnAcc)
        prnterr.append(error)
    
    prnttst = np.array(tstAcc)
    prnttrn = np.array(trnAcc)
    prnterr = np.array(error)
    
    printDF = pd.DataFrame(prnttst)
    csvname = "LmdaTest1Tst.csv"
    printDF.to_csv(csvname)
    
    printDF = pd.DataFrame(prnttrn)
    csvname = "LmdaTest1Trn.csv"
    printDF.to_csv(csvname)
    
    printDF = pd.DataFrame(error)
    csvname = "LmdaTest1Err.csv"
    printDF.to_csv(csvname)
    
    print ('==========Training done')

    # mlp.testAccuracy(trnLst, trnKey)
    # mlp.testAccuracy(tstLst, tstKey)

