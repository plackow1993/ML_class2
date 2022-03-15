import pandas as pd
import numpy as np

from scipy import sparse
from scipy.stats import uniform

#for df in pd.read_csv('training.csv', iterator = True, chunksize = 2000):
    #print(df.dtypes)
#OG_df = pd.read_csv('training.csv', header = None, nrows=1)
#sparse_OG = sparse.csr_matrix(OG_df)



########### reads in the training data

#initializes the sparse matrix by defining final sparse.
#df0 = pd.read_csv('training.csv', header = None, skiprows=0, nrows=1)
#final_sparse = sparse.csr_matrix(df0)


#only for testing skiprows
#dffirst50 = pd.read_csv('training.csv', header = None, skiprows=0, nrows=50)
#dffirst50sparse = sparse.csr_matrix(dffirst50)
#print(dffirst50sparse)
#print(dffirst50)
#for i in range(1, 11999, 857):
#    print(i)
#    dfi = pd.read_csv('training.csv', header = None, skiprows=i, nrows=857)
#    sparse_dfi = sparse.csr_matrix(dfi)
#    final_sparse = sparse.vstack((final_sparse, sparse_dfi))
#
#print(final_sparse)
#sparse.save_npz("training_sparse.npz", final_sparse)






########## reads in the testing data

#initializes the sparse matrix by defining final sparse.
df0 = pd.read_csv('testing.csv', header = None, skiprows=0, nrows=521)
final_sparse = sparse.csr_matrix(df0)


#only for testing skiprows
#dffirst50 = pd.read_csv('testing.csv', header = None, skiprows=6773, nrows=3)
#dffirst50sparse = sparse.csr_matrix(dffirst50)
#print(dffirst50sparse)

for i in range(521, 6773, 1042):
    print(i)
    dfi = pd.read_csv('testing.csv', header = None, skiprows=i, nrows=1042)
    sparse_dfi = sparse.csr_matrix(dfi)
    final_sparse = sparse.vstack((final_sparse, sparse_dfi))

print(final_sparse.shape)
sparse.save_npz("testing_sparse.npz", final_sparse)
