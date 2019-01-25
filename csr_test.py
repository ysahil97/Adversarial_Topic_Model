import numpy as np
from scipy import sparse

a = sparse.random(100,100,format='csr',dtype='float64')

# b = sparse.bsr_matrix(a)
c = sparse.diags(1/a.sum(axis=1).A.ravel())
print(a)
print(c)
print((c@a).todense())
print(a)
print(a/a.sum(axis=1))
x = np.random.rand(10,10)
print(x.dtype)