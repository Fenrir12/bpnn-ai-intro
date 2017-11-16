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
    istep = 28 / rowcells
    jstep = 28 / clmcells

    ftrvec = np.empty([rowcells * clmcells, ])
    for i in range(rowcells):
        for j in range(clmcells):
            strt = i * istep + j * 28 * jstep
            end = strt + istep
            cell = numimage[strt:end]
            for k in range(1, jstep):
                strt = i * istep + j * 28 * jstep + k * 28
                end = strt + istep

                tmp = numimage[strt:end]
                cell = np.concatenate((cell, tmp), axis=0)

            cellcolor = np.mean(cell)

            ftrvec[i * rowcells + j] = cellcolor
    # debug
    # print(ftrvec)
    return ftrvec


def print_cmprsd(ftrvec, rowcells, clmcells):
    istep = (28 / rowcells)
    jstep = (28 / clmcells)

    for j in range(clmcells):
        rowlst = []
        for i in range(rowcells):
            cellclr = str(ftrvec[i * rowcells + j])
            rowlst.append(cellclr)
        print (rowlst)


#########   Explanation  #######################
# reads file containing many hand drawn image files
#########   Input    #######################
# digFile:  file to read digits from16
# rowcells: number of cells in y direction to compress down to
# clmcells: number of cells in x direction to compress down to
#########   Output  #######################
# will output 1D numpy array of length rowcells x clmcells
def read_digit_file(digFile, rowcells, clmcells):
    answerKey = []  # this will hold the actual value of the digit entry
    digitLst = []  # will hold 1D numpy vecs of the data

    trainDF = pd.read_csv(digFile, header=0, index_col=False)

    answerKey = trainDF['label'].tolist()

    # Delete the label from panda dataframe,
    trainDF.drop('label', axis=1)

    # Pop first column of labels
    digitLst = np.array(trainDF.values.tolist())[:,1:]

    if rowcells*clmcells != 784:
        for i in range(len(digitLst)):
            digitLst[i] = compress(digitLst[i], rowcells, clmcells)

    return answerKey, normalize(digitLst)


#########   Explanation  #######################
# agglomerates several songs into a batch
#########   Input    #######################
# digitList:	1-D compressed array of hand drawn digit image data
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

def one_hot_vector(answerKey):
    '''Performs encoding for class labels of training set'''
    onehotvectors = []
    for key in answerKey:
        empty_vec = np.zeros((1, 10))
        empty_vec[0][key] = 1
        onehotvectors.append(empty_vec)
    return onehotvectors

def normalize(digitLst):
    '''Normalize input between 0 and 1'''
    return digitLst/256

if __name__ == "__main__":
    ###############################
    ### Main Program ##############
    ###############################
    trainFile = "smalltrain.csv"

    rowcells = 14
    clmcells = 14

    batchsize = 3

    answerKey, digitLst = read_digit_file(trainFile, rowcells, clmcells)

    # debug
    print("answerkey developed")
    print("read trn", type(answerKey), type(digitLst), type(digitLst[0]))
    print(len(answerKey), len(digitLst), digitLst[0].shape)

    # wait()

    # for i in range(len(digitLst)):

    # 	print "print cmprsd", i
    # 	print_cmprsd(digitLst[i], rowcells, clmcells)
    # print("************Print Cmprsd**********")

    # wait()

    print("************Batch***************")
    for step in range(int(len(digitLst) / batchsize)):
        batch = batchify(digitLst, batchsize, step)

    # debug
    # print(batch)
    # print(batch.shape)
