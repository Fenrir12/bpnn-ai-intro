'''
Coded by Jacob Williams

This file handles compressing, and manipulating song information to
be entered into our Artificial Neural Network
'''

import sys, os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


def wait():
    programPause = raw_input("Press the <ENTER> key to continue...")
    #programPause = input("Press the <ENTER> key to continue...")


#########   Explanation  #######################
# reduces size of hand-drawn image vector
#########   Input   cell #######################
# numimage: 1d numpy vector of grayscale numbers of handrawn digit
# rowcells: number of cells in y direction to compress down to
# clmcells: number of cells in x direction to compress down to
#########   Output  #######################
# will output 1D numpy array of length rowcells x clmcells
def compress(numimage, rowcells, clmcells):
    '''
    this function will input
    '''
    istep = int(28 / rowcells)
    jstep = int(28 / clmcells)

    ftrvec = np.empty([rowcells * clmcells, ])
    for i in range(rowcells):
        for j in range(clmcells):
            strt = i * istep + j * 28 * jstep
            end = strt + istep
            cell = numimage[strt:end]
            for k in range(1, jstep):
                strt = int(i * istep + j * 28 * jstep + k * 28)
                end = int(strt + istep)

                tmp = numimage[strt:end]
                cell = np.concatenate((cell, tmp), axis=0)

            cellcolor = np.mean(cell)

            ftrvec[i * rowcells + j] = cellcolor
    # debug
    # print(ftrvec)
    return ftrvec

#function to print compressed 1d pixel in 2d format
def print_cmprsd(ftrvec, rowcells, clmcells):
    for j in range(clmcells):
        rowlst = []
        for i in range(rowcells):
            cellclr = str(ftrvec[i * rowcells + j])
            rowlst.append(cellclr)
        print (rowlst)

#########   Explanation  #######################
# converts answerkey of digits 0-9 into binary version
#of length ten with position marking answerkey digit
def one_hot_vector(answerKey):
    '''Performs encoding for class labels of training set'''
    onehotvectors = []
    for key in answerKey:
        empty_vec = np.zeros((1, 10), dtype=np.int)
        empty_vec[0][key] = 1
        onehotvectors.append(empty_vec)
    return onehotvectors

#########   Explanation  #######################
# converts digit 0-9 answer to binary len 10 vector
def digtoHotVec(digit):
    empty_vec = np.zeros((1, 10), dtype=np.int)
    empty_vec[0][digit] = 1
    return empty_vec

#########   Explanation  #######################
# reads file containing many hand drawn image files
#########   Input    #######################
# digFile:  file to read digits from16
# rowcells: number of cells in y direction to compress down to
# clmcells: number of cells in x direction to compress down to
#########   Output  #######################
# answerKey: 1D numpy array of length rowcells x clmcells for answerKey
# digitLst:  list of 1D numpy arrays of length rowcells x clmcells, one for 
#            each digit
def read_train(digFile, rowcells, clmcells):
    answerKey = []  # this will hold the actual value of the digit entry
    digitLst = []  # will hold 1D numpy vecs of the data

    trainDF = pd.read_csv(digFile, header=0, index_col=False)

    answerKey = trainDF['label'].tolist()

    # Delete the label from panda dataframe,
    trainDF.drop('label', axis=1)

    digitLst = trainDF.values.tolist()

    for i in range(len(digitLst)):
        answerKey[i] = digtoHotVec(answerKey[i])
        digitLst[i] = compress(digitLst[i], rowcells, clmcells)

    return answerKey, digitLst

#########   Explanation  #######################
# same as read_train, but no answerkey required
def read_test(testFile, rowcells, clmcells):
    digitLst = []  # will hold 1D numpy vecs of the data

    testDF = pd.read_csv(testFile, header=0, index_col=False)

    digitLst = testDF.values.tolist()

    for i in range(len(digitLst)):
        digitLst[i] = compress(digitLst[i], rowcells, clmcells)

    return digitLst

#########   Explanation  #######################
# same as read_train, but specify training size and
# size of testing set, gives answer key to testing to see
# if learning is effective
def read_trn_partial(digFile, rowcells, clmcells, numTrn, numTst):
    answerKey=[]
    digitLst=[]
    
    trnKey = []  # this will hold the actual value of the digit entry
    trnLst = []  # will hold 1D numpy vecs of the data
    
    tstKey = []  # this will hold the actual value of the digit entry
    tstLst = []  # will hold 1D numpy vecs of the data
    
    trainDf = pd.read_csv(digFile, header=0, index_col=False)
    
    '''
    #This function takes a long time to run so decomment these lines if you
    #will be running the same training and testing set over and over
    droprows = range(numTrn+numTst+1, len(trainDf))
    smallDf= trainDf.drop(trainDf.index[droprows])
    smallDf.to_csv("1000trn100tst.csv")
    wait()
    '''
    
    answerKey = trainDf['label'].tolist()

    # Delete the label from panda dataframe,
    trainDf.drop('label', axis=1)

    digitLst = trainDf.values.tolist()

    for i in range(numTrn):
        trnLst.append(compress(digitLst[i], rowcells, clmcells))
        trnKey.append(digtoHotVec(answerKey[i]))
    
    if(numTrn+numTst>len(answerKey)):
        print("Error: numTest + numTrain must be less than number of digits in digit File")
        for i in range(numTrn, len(answerKey)):
            tstLst.append(compress(digitLst[i], rowcells, clmcells))
            tstKey.append(digtoHotVec(answerKey[i]))          
    else:
        for i in range(numTrn, numTrn+numTst):
            tstLst.append(compress(digitLst[i], rowcells, clmcells))
            tstKey.append(digtoHotVec(answerKey[i]))          

    return trnLst, trnKey, tstLst, tstKey

#########   Explanation  #######################
# agglomerates several songs into a batch
#########   Input    #######################
# digitList:	1-D compressed array of hand drawn digit image data
# target_vector: list of 1D vector, 1 per digit in batch.  Each 1d vec,
#                   is of size 10, holding binary value of whether position
#                   is the true digit according to answer key
# size: 	number of hand drawn digit image files to include in batch
# step: 	the batch number to produce, makes sure covers whole data
#########   Output  #######################
# will output numpy array of [size x digit features]

def batchify(digitLst, target_vector, size, step):
    numsteps = int(len(digitLst) / size) - 1

    # Debug
    # print(step*size + size , len(digitLst) )

    if (step * size + size > len(digitLst)):
        print("***ERRROR : Step causes too small of a batch")
        return 0
    else:
        batch = np.array(digitLst[step * size])
        target_b = np.array(target_vector[step * size])
        for i in range(1, size):
            tmp = digitLst[step * size + i]
            tmp2 = target_vector[step * size + i]
            # 1d batch
            # batch = np.concatenate((batch, digitLst[step*size+i]), axis=0)
            # 2d batch
            batch = np.vstack((batch, tmp))
            target_b = np.vstack((target_b, tmp2))
        # debug
        # print("Batch return", type(batch), type(batch[0]))
        if size == 1:
            return [target_b], [batch]
        else:
            return target_b, batch



def normalize(digitLst):
    '''Normalize input between 0 and 1'''
    return digitLst/256


if __name__ == "__main__":
    ###############################
    ### Main Program ##############
    ###############################
    trainFile = "Data/1000trn100tst.csv"

    rowcells = 4
    clmcells = 4
    
    numTrn = 1000
    numTst = 100

    batchsize = 3

    #answerKey, digitLst = read_train(trainFile, rowcells, clmcells)
    #targets = one_hot_vector(answerKey)
    # debug
    #print("read trn", type(answerKey), type(answerKey[0]), type(digitLst), type(digitLst[0]))
    #print(len(answerKey), answerKey[0], len(digitLst), digitLst[0].shape)

    # wait()

    # for i in range(len(digitLst)):

    # 	print "print cmprsd", i
    # 	print_cmprsd(digitLst[i], rowcells, clmcells)
    # print("************Print Cmprsd**********")

    # wait()
    print("************Read Train Partial*********")
    trnLst, trnKey, tstLst, tstKey = read_trn_partial(trainFile, rowcells, clmcells, numTrn, numTst)
    
    print("************Batch***************")
    for step in range(int(len(digitLst) / batchsize)):
        answer_b, batch = batchify(digitLst, answerKey, batchsize, step)

    # debug
    print(batch)
    print(batch.shape)
