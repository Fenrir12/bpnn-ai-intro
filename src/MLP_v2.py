import numpy as np
import matplotlib.pyplot as plt
#import src.numtofeature as ntf
import numtofeature as ntf
import pandas as pd

def wait():
    #programPause = raw_input("Press the <ENTER> key to continue...")
    programPause = input("Press the <ENTER> key to continue...")

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
        self.n_layer = n_layer + 1
        self.W = [0.0] * self.n_layer # Weight of neurons in each layers
        self.A = [0.0] * self.n_layer # Output of neurons in each layers
        self.dC = [0.0] * self.n_layer # Derivative of cost function
        self.dW = [0.0] * self.n_layer # Rate of change of cost function for each neurons
        self.B = [0.0] * self.n_layer # Bias term of each neuron

        # Initialize weights and biases of layers according to depth of network
        if self.n_layer < 1:
            raise ValueError('n_layer should be greater or equal to 1')
        if self.n_layer == 1:
            self.W[0] = np.random.normal(scale=0.1, size=(self.n_input, self.n_output))
            self.B[0] = np.random.normal(scale=0.1, size=(1, self.n_output))
        else:
            self.W[0] = np.random.normal(scale=0.1, size=(self.n_input, self.n_hidden))
            self.B[0] = np.random.normal(scale=0.1, size=(1, self.n_hidden))
            self.W[self.n_layer - 1] = np.random.normal(scale=0.1, size=(self.n_hidden, self.n_output))
            self.B[self.n_layer - 1] = np.random.normal(scale=0.1, size=(1, self.n_output))
            for depth in range(1, self.n_layer - 1):
                self.W[depth] = np.random.normal(scale=0.1, size=(self.n_hidden, self.n_hidden))
                self.B[depth] = np.random.normal(scale=0.1, size=(1, self.n_hidden))
    #jmw
    #########   Explanation  #######################
    #   This function will test a neural network against test data
    #########   Input  #######################
    # trnLst: list of 1D numpy vectors, each vec holds features of 1 digit image
    # trnKey: list of 1d numpy vectors, each vec holds binary representation of 
    #           digits 0-9, position of 1 is the digit.  Corresponds by position with
    #########   Output  #######################
    # Will print out overall accuracy rate, accuracy rate of each digit
    def testAccuracy(self, tstLst, tstKey):
        i=0
        inaccurate=0
        
        for i in range(len(tstLst)):
            predict = self.feedforward(tstLst[i])
            maxpos = np.argmax(predict)
            pretty = np.zeros_like(predict)
            pretty[0][maxpos]= 1
            
            if(maxpos != np.argmax(tstKey[i])):
                '''
                print(tstKey[i])
                print(pretty)
                print(np.argmax(tstKey[i])," ", maxpos)
                '''
                inaccurate += 1.0
        accuracy = 100 - 100 * float(inaccurate/len(tstLst))
        return accuracy
    
    #jmw
    #elementwise version of ReLU equation
    def softplusElement(self, x):
        #approximate of ReLU that you can take a derivative of
        #https://www.quora.com/What-is-special-about-rectifier-neural-units-used-in-NN-learning
        # print(x)
        tmp = 1 + np.exp(x)
        y = np.log(tmp)
        return y

    #jmw
    # elementwise version of derivative of ReLU equation
    def dsoftplusElement(self, x):
        # print("dsplus elm", x)
        y = 1/(1+np.exp(-x))
        return y

    #jmw
    #matix calc for activation function
    def _softplus(self, x):
        # approximate of ReLU that you can take a derivative of
        splus = np.vectorize(self.softplusElement, otypes=[np.float])
        return splus(x)
    
    #jmw
    #matrix calc used in backpropagation
    def _dsoftplus(self, x):
        dsplus = np.vectorize(self.dsoftplusElement, otypes=[np.float])
        return np.exp(x) / (1 + np.exp(x))

    
    def _sigmoid(self, x):
        '''Computes sigmoid of a value'''
        return 1 / (1 + np.exp(-x))

    def _dsigmoid(self, y):
        '''Computes derivative of sigmoid function'''
        return y * (1.0 - y)

    def _softmax(self, batch_o):
        return np.array([np.exp(x) / np.exp(x).sum() for x in batch_o])

    def feedforward(self, inputs):
        '''Fills the activation matrix of the neurons
        and outputs the results at the end for one epoch'''
        for depth in range(self.n_layer):
            # Compute activation of first hidden layer
            if depth == 0:
                self.A[depth] = self._sigmoid(np.dot(inputs, self.W[depth]) + self.B[depth])
            # Activation of every other layer
            else:
                self.A[depth] = self._sigmoid(np.dot(self.A[depth - 1], self.W[depth]) + self.B[depth])
        
        self.A[depth] = self._sigmoid(np.dot(self.A[depth - 1], self.W[depth]) + self.B[depth])
        #print("activation", self.A[depth])

        return self.A[depth]
       
    #jmw
    #same as other but softplus activation, except for final layer
    # this will scale answers back to close to one for error calculation
    def ffpropSplus(self, inputs):
        '''Fills the activation matrix of the neurons
        and outputs the results at the end for one epoch'''
        for depth in range(self.n_layer):
            # Compute activation of first hidden layer
            if depth == 0:
                self.A[depth] = self._softplus(np.dot(inputs, self.W[depth]) + self.B[depth])
            # Activation of every other layer
            else:
                self.A[depth] = self._softplus(np.dot(self.A[depth - 1], self.W[depth]) + self.B[depth])
                
        #activation of last layer??????????
        self.A[depth] = self._sigmoid(np.dot(self.A[depth - 1], self.W[depth]) + self.B[depth])
        #print("activation", self.A[depth])

        return self.A[depth]
    
    #jmw
    #will take the output activation layer and adjust it to proper binary categorical 
    def outActAdjst(self, A):
        outputArray = np.zeros_like(A)
        i=0
        for row in A:
            maxpos = np.argmax(row)
            outputArray[i][maxpos]=1
            i+=1
        
        return outputArray
    
    #jmw
    #will compare the activation layer, and the the label vector and calc
    # the error.  First step Error in backprop
    def outLblError(self, act, lbl, depth):
        adjusted = self.outActAdjst(act)
        #print(act)
        #print("adjusted",adjusted)
        ErrorArray = np.zeros_like(act)
        for i in range(len(act)):
            #print(tmp)
            if(not(np.array_equal(adjusted[i],lbl[i]))):
                ErrorArray[i] = lbl[i] - self.A[depth][i]
                
        return ErrorArray

    #jmw
    #same as your fit, but changed to use soft plus scheme, sigmoid on output layer
    # soft plus on all of the rest
    def fitSp(self, inputs, targets, tstData, tstKey, learning_rate=0.01, n_epochs=200000):
        '''Train the weights of a custom network by computing activations from feedforward
        and then backpropagating the errors after one epoch. Uses simple error as loss function.'''
        ERROR = []
        trnErr = []
        tstErr = []
        batch_size = 10
        numsteps = int(len(inputs) / batch_size) - 1

        print(n_epochs)

        for j in range(n_epochs):
            btchstp = j % numsteps
            targets_b, batch = ntf.batchify(inputs, targets, batch_size, btchstp)
            # Apply feed forward for epoch
            self.ffpropSplus(batch)
            for depth in reversed(range(self.n_layer)):
                # BACKPROPAGATION
                # Squared euclidean distance cost function
                # Compute the error and derivative error of output layers' neurons
                if depth == self.n_layer - 1:
                    # print("shape check", targets_b.shape, self.A[depth].shape)
                    # print(targets_b)
                    # print(self.outActAdjst(self.A[depth]))
                    self.dC[depth] = targets_b - self.A[depth]  # Derivative of the squared euclidean distance
                    # print("output layer error", self.E[depth])
                    self.dW[depth] = np.multiply(self.dC[depth], self._dsigmoid(self.A[depth]))
                # Compute the error and derivative error of hidden layers' neurons
                else:
                    self.dC[depth] = np.dot(self.dW[depth + 1], self.W[depth + 1].T)
                    self.dW[depth] = np.multiply(self.dC[depth], self._dsoftplus(self.A[depth]))

                # Update weights based on contribution of neuron to error
                if depth == 0:
                    self.W[depth] += np.dot(batch.T, self.dW[depth]) * learning_rate
                else:
                    self.W[depth] += np.dot(self.A[depth - 1].T, self.dW[depth]) * learning_rate
                self.B[depth] += np.mean(self.dW[depth], axis=0) * learning_rate
            if (j % 10) == 0:
                print("Error:" + str(np.mean(np.abs(self.dC[self.n_layer - 1]))))
                ERROR.append(np.mean(np.abs(self.dC[-1])))
                trnErr.append(self.testAccuracy(inputs, targets))
                tstErr.append(self.testAccuracy(tstData, tstKey))

                # Show cost function and performance results
                plt.figure(1)
                plt.subplot(211)
                plt.xlabel('Epochs (n)')
                plt.ylabel('Cost function')
                plt.plot(range(0, len(ERROR)), ERROR)
                plt.subplot(212)
                plt.xlabel('Epochs (n)')
                plt.ylabel('Performance')
                plt.plot(range(0, len(trnErr)), trnErr, 'r--')
                plt.plot(range(0, len(tstErr)), tstErr, 'b')
                plt.pause(0.000001)

            if((j+1)%100==0):
                printlst = []
                printlst.append(ERROR)
                printlst.append(trnErr)
                printlst.append(tstErr)
                printlst = np.array(printlst)
                printDF= pd.DataFrame(printlst)
                csvname = "ErrRcrdFtrs"+str(len(inputs[0]))+"Hdn"+str(self.n_hidden)+\
                           "Trn"+str(len(targets))+"Tst"+str(len(tstData))+"It"+str(j)+".csv"
                printDF.to_csv(csvname)
        #plt.show()
            

    def fitSg(self, inputs, targets, tstData, tstKey, learning_rate=0.01, n_epochs=20000):
        '''Train the weights of a custom network by computing activations from feedforward
        and then backpropagating the errors after one epoch. Uses simple error as loss function.'''
        ERROR = []
        trnErr = []
        tstErr = []
        batch_size = 10
        numsteps = int(len(inputs) / batch_size) - 1

        print(n_epochs)

        for j in range(n_epochs):
            btchstp = j % numsteps
            targets_b, batch = ntf.batchify(inputs, targets, batch_size, btchstp)
            # Apply feed forward for epoch
            self.feedforward(batch)
            for depth in reversed(range(self.n_layer)):
                # BACKPROPAGATION
                # Squared euclidean distance cost function
                # Compute the error and derivative error of output layers' neurons
                if depth == self.n_layer - 1:
                    #print("shape check", targets_b.shape, self.A[depth].shape)
                    self.dC[depth] = targets_b - self.A[depth]  # Derivative of the squared euclidean distance
                    self.dW[depth] = np.multiply(self.dC[depth], self._dsigmoid(self.A[depth])) # Haddamard product
                # Compute the error and derivative error of hidden layers' neurons
                else:
                    self.dC[depth] = np.dot(self.dW[depth + 1], self.W[depth + 1].T)
                    self.dW[depth] = np.multiply(self.dC[depth], self._dsigmoid(self.A[depth])) # Haddamard product

                # Update weights based on contribution of neuron to error
                # if input layer
                if depth == 0:
                    self.W[depth] += np.dot(batch.T, self.dW[depth]) * learning_rate
                else:
                    self.W[depth] += np.dot(self.A[depth - 1].T, self.dW[depth]) * learning_rate
                self.B[depth] += np.mean(self.dW[depth], axis=0) * learning_rate
            if (j % 5) == 0:
                '''
                print("Error:" + str(np.mean(np.abs(self.E[self.n_layer - 1]))))
                '''
                ERROR.append(np.mean(np.abs(self.dC[-1])))
                trnErr.append(self.testAccuracy(inputs, targets))
                tstErr.append(self.testAccuracy(tstData, tstKey))
                
                #print("Trn Error,", trnErr,"Tst Error,", tstErr)
            if((j+1)%10==0):
                printlst = []
                printlst.append(ERROR)
                printlst.append(trnErr)
                printlst.append(tstErr)
                printlst = np.array(printlst)
                printDF= pd.DataFrame(printlst)
                csvname = "ErrRcrdFtrs"+str(len(inputs[0]))+"Hdn"+str(self.n_hidden)+\
                           "Trn"+str(len(targets))+"Tst"+str(len(tstData))+"It"+str(j)+".csv"
                printDF.to_csv(csvname)

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

        plt.show()

    #jmw
    ###candidate for deletion, all it does is call feed forward
    def predict(self, i):
        '''Returns the output of the network given an input matching input neurons'''
        #i is 1d vector of input data
        return self.feedforward(i)

    

if __name__ == "__main__":
    mlp = MLP(784, 100, 10, 1)
    # Test XOR, AND, OR and NOR inputs and targets
    # inputs = np.array([[[0, 0], [0, 1], [1, 0], [1, 1]],
    #                    [[0, 0], [0, 1], [1, 0], [1, 1]],
    #                    [[0, 0], [0, 1], [1, 0], [1, 1]],
    #                    [[0, 0], [0, 1], [1, 0], [1, 1]]])
    # targets = np.array([[[0, 0, 0, 0], [1, 0, 0, 0], [1, 0, 0, 0], [0, 0, 0, 0]],
    #                     [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 1, 0, 0]],
    #                     [[0, 0, 0, 0], [0, 0, 1, 0], [0, 0, 1, 0], [0, 0, 1, 0]],
    #                     [[0, 0, 0, 1], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]]])
    print ('==========Extracting train set')
    trnLst, trnKey, tstLst, tstKey = \
    ntf.read_trn_partial('data/5000trn500tst.csv', 28, 28, 5000, 500)

    #print(trnLst[0].shape)
    print ('==========Training...')
    mlp.fitSp(trnLst, trnKey, tstLst, tstKey, learning_rate=0.01, n_epochs=20000)
    print ('==========Training done')
    
    #mlp.testAccuracy(trnLst, trnKey)
    #mlp.testAccuracy(tstLst, tstKey)

