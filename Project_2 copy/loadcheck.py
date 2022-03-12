#This loads in the final training sparse matrix for us to use now freely. Another reading file will have to be used to read in the test data, or just modify categorization and load.py
import numpy as np
import pandas as pd
from scipy import sparse
from scipy.sparse import linalg
from sklearn.model_selection import train_test_split
import sys
np.set_printoptions(threshold=sys.maxsize)

#read in vocabular and labels text files. Can convert to sparse if needed.
vocabulary = pd.read_csv("vocabulary.txt", header = None)
labels = pd.read_csv("newsgrouplabels.txt", header = None)

#load in saved sparse training data from categorization and load.py scripts.
training_data = sparse.load_npz('final_training_sparse.npz')

#helpful for understanding the structure of a sparse matrix, but not needed for our code.
#print(training_data[:,61188])
#print("The class of the word", vocabulary.loc[61187, 0], 'is')
#print(labels.loc[training_data[6413, 61189]-1, 0])

#quick construction of the P(Yk) MLE estimation. PY_MLE is the estimated Maximum likelihood error function and is a sparse vector of size 20x1 (kx1). This is training k values, i didnt utilize SUM P = 1.
#20% split of the training data into training and validation.
training_X, test_X, training_Y, test_Y = train_test_split(training_data[:,1:-1], training_data[:,-1], test_size = 0.2, random_state = 40)


######construction of the MAP P(X|Y).

#beta will be 1/abs_V. We can change this and incorporate into a loop for more betas.
abs_V = vocabulary.shape[0]
beta = 1/abs_V


#this retains the position of each class in the training set, to use to sum up all xi in each class.
total_length = 0
#converting a list of betas into one sparse matrix for sparse matrix operations (addition is not supported elementwise as far as i can tell.
add_factor = [beta]*len(vocabulary)
add_factor = sparse.csr_matrix(add_factor)

for y in labels.index:
    Ypositions = []

    for i in range(1, training_Y.indptr.shape[0]):
        if (training_Y==(y+1)).indptr[i-1] == (training_Y==(y+1)).indptr[i]-1:
            Ypositions.append(i-1)
    total_length += len(Ypositions)
    #sums ALL xis, and gives a 1xlen(vocabulary) list of sums for class y+1
    sum_xi = training_X[Ypositions, :].sum(axis=0)
    #sums ALL words (aka sum of sums of xis) in the yk class
    sum_yk = training_X[Ypositions, :].sum()
    sum_xi = sparse.csr_matrix(sum_xi)
    
    #this is the P(Xi, Yk) row for each x_i with a dirichlet MAP estimation.
    xi_dirichlet = (sum_xi+add_factor)/(sum_yk+beta*abs_V)
    #added this in just to check if it adds to 1. It adds to very near 1 for all cases.
    #print(((sum_xi+add_factor)/(sum_yk+beta*abs_V)).sum())
    
    #basecase to initialize our sparse matrix
    if y == 0:
        MAP_estimate = xi_dirichlet
    else:
        MAP_estimate = sparse.vstack((MAP_estimate, xi_dirichlet))

#this checks that the MAP matrix is indeed 20 x 61188, classes(yk) by words(xi)
#print(MAP_estimate.shape)
#this changes MAP_estimate into its log for training.
MAP_estimate.data = np.log(MAP_estimate.data)
print(MAP_estimate.shape)
        

vector_totals = []
for y in labels.index:
    totals = 0
    for x in range(0, training_X.shape[0]):
        if training_Y[x]==(y+1):
            totals += 1
    vector_totals.append(totals)
vector_totals = sparse.csr_matrix(vector_totals)
PY_MAE = vector_totals/sparse.csr_matrix.sum(vector_totals)
PY_MAE.data = np.log(PY_MAE.data)
print(PY_MAE.shape)


#argmax should be a vector based function of the form log(P(Y)_MAE)+X_i*P(X|Y)_MAP^T, and result in a vector of size 1x20.
#Test_X is a 2400 x 61188 matrix. so take each ROW and use it to make a prediction. Those rows are 1x61188.
predictions = []
for x in range(0,test_X.shape[0]):
    arg_search = PY_MAE + test_X[x, :]*MAP_estimate.transpose()
    predictions.append(arg_search.argmax(axis = 1)+1)

count_correct = 0
for p in range(0,len(predictions)):
    if predictions[p] == test_Y[p, 0]:
        count_correct += 1
        
print(count_correct)

    
