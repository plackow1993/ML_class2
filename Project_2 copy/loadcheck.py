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



#read in vocabulary and labels text files. Can convert to sparse if needed.
vocabulary = pd.read_csv("vocabulary.txt", header = None)
labels = pd.read_csv("newsgrouplabels.txt", header = None)


#load in saved sparse training data from categorization and load.py scripts.
training_data = sparse.load_npz('final_training_sparse.npz')
testing_data = sparse.load_npz('final_testing_sparse.npz')
testing_data = testing_data[:, 1:61189]



#quick construction of the P(Yk) MLE estimation. PY_MLE is the estimated Maximum likelihood error function and is a sparse vector of size 20x1 (kx1). This is training k values, i didnt utilize SUM P = 1.
#20% split of the training data into training and validation, test_size = 0.2.
training_X, test_X, training_Y, test_Y = train_test_split(training_data[:,1:-1], training_data[:,-1], test_size = 1/12000, random_state = 40)

#########################################
#THIS is only for testing time expected for program to finish by step size of beta.
#beta_length = []
#for B in range(1,100002,2000):
#    print(B)
#    beta_length.append(B/100000)
#print(beta_length[0], beta_length[-1])
#print(len(beta_length))
#quit()
#########################################

####construction of MAE for P(Y)
#construct this first, since this is independent of beta. Saves a ton of time when looping over different beta.
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

######construction of the MAP P(X|Y).

#beta will be 1/abs_V. We can change this and incorporate into a loop for more betas. (this instantiates the beta to be the size of 1/abs_V. Ill comment this out and make a list of betas to loop the training function over. The accuracy is on the validation set from the split above.
abs_V = vocabulary.shape[0]
#beta = 1/abs_V
accuracy_probs = []
beta_vector = []


#this retains the position of each class in the training set, to use to sum up all xi in each class.
total_length = 0

#You get 51 data points by using range(0,100001, 200). Use for checking best beta
step_count = 0

#Use "for B in Beta:" that is a list for finding best beta that trains the model (beta = 0.006), with 20% of training used for validation
#Use "for B in [best_beta]:" for testing on testing data for kaggle submission
#### TS
#Beta = [0.00001, 0.00005, 0.0001, 0.0005, 0.001, 0.005,  0.01, 0.05,  0.5, 1]
#### BB / CM
best_beta = 0.0045
Beta = [best_beta]
for B in Beta:
    #use for single element list or for a list with values in place.
    beta = B
    
    
    print(beta)
    beta_vector.append(beta)
    
    #This was used to pick the positions of class values from the sparse matrix training_Y
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
        #converting a list of betas into one sparse matrix for sparse matrix operations (addition is not supported elementwise as far as i can tell.
        add_factor = [beta]*len(vocabulary)
        add_factor = sparse.csr_matrix(add_factor)
        xi_dirichlet = (sum_xi+add_factor)/(sum_yk+beta*abs_V)
        
        #added this in just to check if it adds to 1. It adds to very near 1 for all cases.
        #print(((sum_xi+add_factor)/(sum_yk+beta*abs_V)).sum())
        
        #basecase to initialize our sparse matrix
        if y == 0:
            MAP_estimate = xi_dirichlet
        else:
            MAP_estimate = sparse.vstack((MAP_estimate, xi_dirichlet))
        print(y)
    #this checks that the MAP matrix is indeed 20 x 61188, classes(yk) by words(xi)
    #print(MAP_estimate.shape)
    
    #this changes MAP_estimate into its log for training.
    MAP_estimate.data = np.log(MAP_estimate.data)
    
            




    #argmax should be a vector based function of the form log(P(Y)_MAE)+X_i*P(X|Y)_MAP^T, and result in a vector of size 1x20.
    #Test_X is a 2400 x 61188 matrix. so take each ROW and use it to make a prediction. Those rows are 1x61188.
    
    #testing_data is the test data supplied on kaggle. it is a 6773 x 61188 matrix.
    predictions = []
    
    
    
####### TS/CM validation loop, keep on for Beta =list (or when using test_X in CM)
#    for x in range(0,test_X.shape[0]):
#        arg_search = PY_MAE + test_X[x, :]*MAP_estimate.transpose()
#        predictions.append(arg_search.argmax(axis = 1)+1)
#    count_correct = 0
#    for p in range(0,len(predictions)):
#        if predictions[p] == test_Y[p, 0]:
#            count_correct += 1
#    accuracy_probs.append(count_correct/test_X.shape[0])
        
        
        
###### BB testing data loop, keep on for B in [best_beta]

    for x in range(0, testing_data.shape[0]):
        arg_search = PY_MAE + testing_data[x, :]*MAP_estimate.transpose()
        predictions.append(arg_search.argmax(axis = 1)+1)
        print(x)

    for p in range(0,len(predictions)):
        predictions[p] = predictions[p].item(0,0)
        print(p)
    print(predictions)
 
 ######## CM Confusion matrix loop, want on when you plan to make a confusion matrix. Otherwise its just adding time to compile.
 
    #confusion matrix Cij, where the rows are the classes as predicted and the columns are actual values. The counter will add if i = j
#    confusion = pd.DataFrame(data = 0, index = range(1,21), columns = range(1,21))
#    for p in range(0,len(predictions)):
#        predictions[p] = predictions[p].item(0,0)
#        print(p)
#    print(predictions)
#    for p in range(0,len(predictions)):
#        if predictions[p] == test_Y[p,0]:
#            confusion.loc[test_Y[p,0], test_Y[p,0]] = confusion.loc[test_Y[p,0], test_Y[p,0]] + 1
#        else:
#            confusion.loc[predictions[p], test_Y[p,0]] = confusion.loc[predictions[p], test_Y[p,0]] + 1
#
#    print(confusion)
#
    step_count += 1
    
    
    
    print(step_count)

end_time = time.time()
##### TS only for Beta = list:
#max_acc = max(accuracy_probs)
#print(max_acc)
#
##max_beta = beta_vector.index(max_acc)
#plt.xscale("log")
#plt.plot(beta_vector, accuracy_probs)
#plt.xlabel("Beta")
#plt.ylabel("Accuracy")
#plt.title("Beta vs. Accuracy for Naive Bayes")
#plt.savefig("Beta_vs_Accuracy_for_Naive_Bayes_smaller.png")
#plt.show()



print("this takes", end_time-start_time, "seconds to run")
    
    
##BB only use for when you are NOT making the confusion matrix or within the beta list to find the best beta. Saves prediction to a file called sumbission.csv.
id_list = []
for i in range(0,testing_data.shape[0]):
    id_list.append(i+12001)
pred_df = pd.DataFrame({"id": id_list, "class": predictions})
print(pred_df)
pred_df.to_csv('submission.csv', index=False)



##saving confusion matrix to a file.
#confusion.to_csv('confusion_NB.csv')
