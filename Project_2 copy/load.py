import numpy as np
import pandas as pd
from  scipy import sparse

#f=sparse.load_npz('training_sparse.npz')
#final = pd.read_csv('training.csv', header = None, skiprows=11999, nrows=1)
#final_sparse = sparse.csr_matrix(final)
#final_sparse = sparse.vstack((f, final_sparse))
#sparse.save_npz("final_training_sparse.npz", final_sparse)

f=sparse.load_npz('testing_sparse.npz')
final = pd.read_csv('testing.csv', header = None, skiprows=6773, nrows=1)
final_sparse = sparse.csr_matrix(final)
final_sparse = sparse.vstack((f, final_sparse))
sparse.save_npz("final_testing_sparse.npz", final_sparse)
