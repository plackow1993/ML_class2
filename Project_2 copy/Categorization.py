import pandas as pd
import numpy as np

from scipy import sparse
from scipy.stats import uniform

#for df in pd.read_csv('training.csv', iterator = True, chunksize = 2000):
    #print(df.dtypes)
#OG_df = pd.read_csv('training.csv', header = None, nrows=1)
#sparse_OG = sparse.csr_matrix(OG_df)
#df0 = pd.read_csv('training.csv', header = None, skiprows=0, nrows=1)
#final_sparse = sparse.csr_matrix(df0)

dffirst50 = pd.read_csv('training.csv', header = None, skiprows=0, nrows=50)
dffirst50sparse = sparse.csr_matrix(dffirst50)
print(dffirst50sparse)
print(dffirst50)
quit()
for i in range(1, 11999, 857):
    print(i)
    dfi = pd.read_csv('training.csv', header = None, skiprows=i, nrows=857)
    sparse_dfi = sparse.csr_matrix(dfi)
    final_sparse = sparse.vstack((final_sparse, sparse_dfi))
    
print(final_sparse)
sparse.save_npz("training_sparse.npz", final_sparse)
