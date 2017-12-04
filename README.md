# bpnn-ai-intro
This repository contains the latest versions for a custom backpropagation artificial neural network.

# Getting started
1. cd into your workspace
2. type the following commands in your terminal: 
```
git clone https://github.com/Fenrir12/bpnn-ai-intro.git 
cd bpnn-ai-intro
```
3. Make sure python dependencies are installed

Note : The base code will train a model on a small dataset of 1000 training samples and 100 testing samples.
       To change the dataset to train with, go into src/MLP_v5.py at line 319 and change .csv file to use custom dataset and parameters 4 and 5 to fit your training and testing set sizes.
       
4. Train the neural network on this dataset and waituntil the end tog et the results.
By uncommenting lines 295 to 306, you can get graphic visualization of the learning progresses
```
python MLP_v5.py
```
5. You can reuse the class MLP and its methods to train your own neural network and customize its performances with the following implemented tools and hyperparameters:
* Dropout
* Regularization
* Momentum
* Batch size
* Learning rate
* Activation functions :
  * Sigmoid
  * Softplus
  * Softmax
* Cost functions :
  * Squared euclidean distance
  * Cross entropy
       
