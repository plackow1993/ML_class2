#Function to calculate entropy of the words in each class.

#This loads in the final training sparse matrix for us to use now freely. Another reading file will have to be used to read in the test data, or just modify categorization and load.py
import numpy as np
import pandas as pd
from scipy import sparse
from scipy.sparse import linalg
from scipy.sparse import find
from sklearn.model_selection import train_test_split
import sys
import time
import matplotlib.pyplot as plt
from sklearn.preprocessing import normalize
import math
start_time = time.time()
np.set_printoptions(threshold=sys.maxsize)


#read in vocabular and labels text files. Can convert to sparse if needed.
vocabulary = pd.read_csv("vocabulary.txt", header = None)
labels = pd.read_csv("newsgrouplabels.txt", header = None)

vocab_list = []
for i in range(0,61188):
    vocab_list.append(vocabulary.iloc[i,0])


#load in saved sparse training data from categorization and load.py scripts.
training_data = sparse.load_npz('final_training_sparse.npz')

random_seed = 40
np.random.seed(random_seed)

#20% split of the training data into training and validation.
training_X, test_X, training_Y, test_Y = train_test_split(training_data[:,1:-1], training_data[:,-1], test_size = 0.02, random_state = random_seed)

#print(training_X.sum(axis = 1))

#Entropy

#class_fractions compute the ratios of fractions for each of class yk in each subframe
def classFractions(subclass_frame):
    fraction_class_list = []
    for y in range(1,21):
        class_counts = 0
        for k in range(0,subclass_frame.shape[0]):
            if subclass_frame[k,0] == y:
                class_counts += 1
        #fraction_class_list.append(class_counts/subclass_frame.shape[0])
        fraction_class_list.append(class_counts)
    Sparse = sparse.csr_matrix(fraction_class_list)
    return Sparse

#want subframe to split into yes's (the word exists) and no's (the word doesnt exist) for each example.
#This then gives a subframe of all yes examples for each x_i word. The number of nos, then is the opposite of these (opposite = 9600 - A)
def subframes(training_matrix, X_i):
    A= training_matrix[:,X_i].nonzero()[0]
    B = sparse.csr_matrix(A)
    return B

    
#### This gives the number of yesses (if the word appears in a class) per class (the opposite of this is the nos, the word doesn't exist in any of the examples for that class)
for n in range(0,61188):
    X = subframes(training_data[:,1:-1], n)
    #YES = classFractions(Y)
    #NO = classFractions(N)
    #print(YES)
    A = X[0,:].transpose()
    A_list = []
    for a in range(0,A.shape[0]):
        A_list.append(A[a,0])
    training_frame = training_data[:,-1][A_list,:]
    if n == 0:
        yes_frame = classFractions(training_frame).transpose()
    else:
        yes_frame = sparse.hstack((yes_frame,classFractions(training_frame).transpose()))
    print(n)
print(yes_frame)
print(yes_frame.shape)
yes_size = yes_frame.sum(axis = 0)
print(yes_size)
#only need to save the yes frame once
sparse.save_npz('yes_frame_entropy_problem6.npz', yes_frame)
end_time = time.time()
print("all x_is exs take", end_time-start_time, "seconds")
quit()



