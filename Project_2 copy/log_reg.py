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
from sklearn.preprocessing import normalize
start_time = time.time()
np.set_printoptions(threshold=sys.maxsize)


#read in vocabular and labels text files. Can convert to sparse if needed.
vocabulary = pd.read_csv("vocabulary.txt", header = None)
labels = pd.read_csv("newsgrouplabels.txt", header = None)

#load in saved sparse training data from categorization and load.py scripts.
training_data = sparse.load_npz('final_training_sparse.npz')
testing_data = sparse.load_npz('final_testing_sparse.npz')
testing_data = testing_data[:, 1:61189]

random_seed = 40
np.random.seed(random_seed)

#20% split of the training data into training and validation.
training_X, test_X, training_Y, test_Y = train_test_split(training_data[:,1:-1], training_data[:,-1], test_size = 0.2, random_state = random_seed)

#normalization of training_X by attribute for overflow correction. So SUMj(X_i,j)=1
training_X = normalize(training_X, norm='l1', axis=0)


###### Calculating the delta function.
#this is a 20 x training_Y.shape[0] (9600) size matrix. Where if example m is class k, then delta(k,m) = 1. Else, delta(k,m) = 0.

delta = pd.DataFrame(data = 0, index = range(1,21), columns = range(1, training_Y.shape[0]+1))
for m in range(0,training_Y.shape[0]):
    for k in range(1,21):
        if training_Y[m,0] == k:
            delta.iloc[k-1,m] = 1
delta = sparse.csr_matrix(delta)


########## Initialize a weight vector with random small weights
#makes a dataframe of size 20x61189 (kx(n+1)), w0 should be multiplied by 1 in the X^T matrix. So X_T should be 61189 x 9600, with the first column being 1s.
W = pd.DataFrame(data = np.random.rand(20, training_X.shape[1]+1), index = range(1,21), columns = range(0, training_X.shape[1]+1) )



######## Create X_T matrix with an added column of 1s.
#training_X is a 9600 x 61188 matrix, or m x n
#X_T is a 61188x9600 matrix or n x m (itll be sparse)

#ones is a row matrix
ones_list = [1]*training_X.shape[0]
ones = sparse.csr_matrix(ones_list, [(1, training_X.shape[0])])
ones = ones.transpose()
training_X = sparse.hstack((ones, training_X))



one_list = [1]*test_X.shape[0]
one = sparse.csr_matrix(one_list, [(1, test_X.shape[0])])
one = one.transpose()
test_X = sparse.hstack((one, test_X))
#X_T is the transpose of the training matrix + 1s for w0 term. It is a 61188+1 x (m) matrix
X_T = training_X.transpose()

######### Make W sparse an multiply W*X_T to make a 20 x 9600 matrix.
W_sparse = sparse.csr_matrix(W)

    

#hyperparameters lists
#step limit for testing algorithm. Step limit as a vector takes a long time. So for each step limit: 500, 1000, 5000, 10000, chosen in answering question 3, i will alter the etal and lambl lists.

#step_limitl = [2,3,4]
step_limitl = [5000]

#Eg: for step_limit = 500, 1000: etal and lambl are length 10 vectors from 0.01 to 0.001. These do not take so long. But for step_limit = 5000, these vectors should be about length 5-8. To account for the more steps taken. Edit: just going to make them all 4. So im only doing this 16 times instead of 100

etal = []
lambl = []
for i in range(1,11,3):
    etal.append(i/1000)
    lambl.append(i/1000)
    

## Specialized etal and lambl for step_limit = 10000
#etal = [.001, .005, .01]
#lambl = etal


# W_sparse needs to reset back to W_Sparse everytime we run a step_limit, eta, lambda run.
frame_list = []
for step_limit in step_limitl:
    step_frame = pd.DataFrame(data=0, index = etal, columns = lambl)
    for eta in etal:
        for lamb in lambl:
            print('code is at eta and lambda of')
            print(eta, lamb)
            W_sparse = sparse.csr_matrix(W)
            step = 0
            while step < step_limit:
                PY_unnormal = W_sparse@X_T
                #takes the exponent of the product of W_sparse*X_T
                PY_unnormal.data = np.exp(PY_unnormal.data)

                #replaces the last row with 1s for class 20
                PY_unnormal[-1,:] = ones.transpose()
                #normalizes PY_unnormal (using PY_normal.sum(axis=0) shows totals of 1 for all weights)
                PY_normal = normalize(PY_unnormal, norm='l1', axis=0)
                ########### Updating the Initial W with the update equation and regularization.
                E = delta - PY_normal
                W_sparse = W_sparse + eta*(E@training_X - lamb*W_sparse)
                resid = (E.data)**2
                print(resid.sum())
                step += 1
                print(step)
                
                
            ########### Testing accuracy for the test data
            unnormal = W_sparse@test_X.transpose()
            unnormal[-1, :] = one.transpose()
            normal = normalize(unnormal, norm = 'l1', axis =0)
            call = []
            for att in range(0, test_X.shape[0]):
                call_vect = normal[:,att]
                call.append((call_vect.argmax(axis = 0)+1).item(0,0))
            count_correct = 0
            for i in range(0,len(call)):
                if call[i] == test_Y[i]:
                    count_correct += 1
            step_frame.loc[eta, lamb] = count_correct/2400
                #only necessary for playing with updating parameters once we get to a certain error.
            #    resid = (E.data)**2
            #    if resid.sum() < 3000:
            #        eta = 0.02
            #        lamb = 0.02
            #
            #    if resid.sum() < 100:
            #        break
            #    print(resid.sum())
                
    frame_list.append(step_frame)
    print(step_limit)
print(len(frame_list))
for i in range(0,len(frame_list)):
    name_string_1 = str(step_limitl[i])
    name_string = "step_limit_" + name_string_1 + "_errors.csv"
    print(name_string)
    frame_list[i].to_csv(name_string)
#The chose class of an example Z (1x61188) vector = argmax (W_sparse*ZT). W_sparse = matrix of size kx61189. So W_sparse*Z^T = kx1. Which class_k gives the largest W_sparse*Z^T.
#Choose test example.

#for x in range(1,50):
#    exp_test = test_X[x, :]
#    exp_test_one = sparse.hstack((1,exp_test))
#    class_probs=W_sparse*exp_test_one.transpose()
#    print(class_probs)
#    print(class_probs.argmax(axis = 0)+1)
#    print(test_Y[x])





######### Testing class calls with initial W_sparse. Checking to see if submission passes kaggle requirements. Only need if making prediction submission for kaggle. Comment out for confusion matrix and loops for eta and lambda and step_limit
#id_list = []
#for i in range(0,testing_data.shape[0]):
#    id_list.append(i+12001)
#
#
##preparing test matrix for later accuracy calculations (need to add a 1 for W0
#ones_list = [1]*testing_data.shape[0]
#ones = sparse.csr_matrix(ones_list, [(1, testing_data.shape[0])])
#ones = ones.transpose()
#testing_data = sparse.hstack((ones, testing_data))
#
##adjust W_sparse so that W_K = 1-Sum(W_1 to W_K-1)
#call_unnormal = W_sparse@testing_data.transpose()
#call_unnormal[-1, :] = ones.transpose()
#call_normal = normalize(call_unnormal, norm = 'l1', axis =0)
#class_call = []
#data_frame_calls = []
#for att in range(0, testing_data.shape[0]):
#    call_vec = call_normal[:,att]
#    class_call.append(call_vec.argmax(axis = 0)+1)
#    data_frame_calls.append(class_call[att].item(0,0))
#pred_df = pd.DataFrame({"id": id_list, "class": data_frame_calls})
#pred_df.to_csv('logistic_submission.csv', index =False)
