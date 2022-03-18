#Logistic Regression Script


#This loads in the final training sparse matrix for us to use now freely. Another reading file will have to be used to read in the test data, or just modify categorization and load.py
import numpy as np
import pandas as pd
from scipy import sparse
from scipy.sparse import linalg
from sklearn.model_selection import train_test_split
import sys
import time
import matplotlib.pyplot as plt
start_time = time.time()
np.set_printoptions(threshold=sys.maxsize)



#read in vocabular and labels text files. Can convert to sparse if needed.
vocabulary = pd.read_csv("vocabulary.txt", header = None)
labels = pd.read_csv("newsgrouplabels.txt", header = None)

#load in saved sparse training data from categorization and load.py scripts.
training_data = sparse.load_npz('final_training_sparse.npz')
testing_data = sparse.load_npz('final_testing_sparse.npz')
testing_data = testing_data[:, 1:61189]


#helpful for understanding the structure of a sparse matrix, but not needed for our code.
#print(training_data[:,61188])
#print("The class of the word", vocabulary.loc[61187, 0], 'is')
#print(labels.loc[training_data[6413, 61189]-1, 0])

#quick construction of the P(Yk) MLE estimation. PY_MLE is the estimated Maximum likelihood error function and is a sparse vector of size 20x1 (kx1). This is training k values, i didnt utilize SUM P = 1.
#20% split of the training data into training and validation.
training_X, test_X, training_Y, test_Y = train_test_split(training_data[:,1:-1], training_data[:,-1], test_size = 0.2, random_state = 40)


###### Calculating the delta function.
#this is a 20 x training_Y.shape[0] (9600) size matrix. Where if example m is class k, then delta(k,m) = 1. Else, delta(k,m) = 0.

delta = pd.DataFrame(data = 0, index = range(1,21), columns = range(1, training_Y.shape[0]+1))
for m in range(0,training_Y.shape[0]):
    for k in range(1,21):
        if training_Y[m,0] == k:
            delta.iloc[k-1,m] = 1
            
print(delta)
