#This is to use the already made sparse matrices of yes counts per x_i

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



#opens the matrix with the yes counts of each class per word and the no counts per word.
yes_frame = sparse.load_npz('yes_frame_entropy_problem6.npz')
Yes_total = yes_frame.sum(axis=0)
Yes_fraction = Yes_total/9600
Yes_normal = normalize(yes_frame, norm='l1', axis=0)

# no counts for entropy calculation
no_frame = sparse.load_npz('no_frame_entropy_problem6.npz')
No_total = no_frame.sum(axis = 0)
No_normal = normalize(no_frame, norm='l1', axis=0)
No_fraction = No_total/9600
#print(np.multiply(Yes_fraction,No_fraction)[0,1])
entropy_yes = sparse.csr_matrix((1,61188))
entropy_no = sparse.csr_matrix((1,61188))
for x in range(0,61188):
    yes_sparse = Yes_normal[np.nonzero(Yes_normal[:,x])[0], x]
    yes_sparse.data = yes_sparse.data*np.log2(yes_sparse.data)
    entropy_yes[0,x] = -yes_sparse.sum()*Yes_fraction[0,x]
    no_sparse = No_normal[np.nonzero(No_normal[:,x])[0], x]
    no_sparse.data = no_sparse.data*np.log2(no_sparse.data)
    entropy_no[0,x] = -no_sparse.sum()*No_fraction[0,x]
    print(x)
    
#the entropy matrices are the entropy parts for each x_i. Multiply these by a factor of the sums/normalized for each yes and no matrix. (this is done within the loop above since yes/no_fractions contains these values per x_i
#print(entropy_yes.shape)
#print(entropy_no.shape)

entropy_total = entropy_yes + entropy_no

#Since this is entropy, (not information gain) we want the least entropy (instead of most information gain) since E(Main_data) is the same for all IG calculations.
index_list = []
for i in range(1,101):
    print(i)
    m =entropy_total.min(axis =1).tocsr()
    min_value_index = (entropy_total==m[0,0]).nonzero()[1][0]
    index_list.append(vocabulary.iloc[min_value_index,0])
    #all values are about 0-4, so if you change the value you just found to 100, it will not be found to be the minimum. Can change this to higher numbers if entropy values get higher.
    entropy_total[0,min_value_index] = 100
print("the top 100 words that will classify the dataset are (in order):")
print(index_list)
quit()



### Not necessary to run this. But it was used to make the matrices for the no frame and the yes frame. If you make them first it saves time in calculating entropy.


#random_seed = 40
#np.random.seed(random_seed)
#
##20% split of the training data into training and validation.
#training_X, test_X, training_Y, test_Y = train_test_split(training_data[:,1:-1], training_data[:,-1], test_size = 0.2, random_state = random_seed)
#
##class_fractions compute the ratios of fractions for each of class yk in each subframe Can use for information gain, but since this factor is included in every example, its unnecessary for ranking the words
#def classFractions(subclass_frame):
#    fraction_class_list = []
#    for y in range(1,21):
#        class_counts = 0
#        for k in range(0,subclass_frame.shape[0]):
#            if subclass_frame[k,0] == y:
#                class_counts += 1
#        #fraction_class_list.append(class_counts/subclass_frame.shape[0])
#        fraction_class_list.append(class_counts)
#    Sparse = sparse.csr_matrix(fraction_class_list)
#    return Sparse
##Y_Frame = classFractions(training_Y)
##print(Y_Frame)
##all the values in Y_frame are the counts that each class appears.
#
##only use once to print
##start_frame = Y_Frame.transpose()
##for x in range(1,61188):
##    start_frame = sparse.hstack((start_frame, Y_Frame.transpose()))
##print(start_frame)
##no_frame = start_frame - yes_frame
##sparse.save_npz('no_frame_entropy_problem6.npz', no_frame)
##print(no_frame.sum(axis=0))

