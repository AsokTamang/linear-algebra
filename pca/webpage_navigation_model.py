import numpy as np
import matplotlib.pyplot as plt
import scipy.sparse.linalg

P = np.array([

    [0, 0.75, 0.35, 0.25, 0.85],
    [0.15, 0, 0.35, 0.25, 0.05],
    [0.15, 0.15, 0, 0.25, 0.05],
    [0.15, 0.05, 0.05, 0, 0.05],
    [0.55, 0.05, 0.25, 0.25, 0]
])
#here the matrix p shows the probability of moving from one page to another
#so pij means the probability of moving from page j to page i ,
#as p03=0.25 means the probability of moving from page 4 to page 1 is 0.25

X0 = np.array([[0], [0], [0], [1], [0]])


#Multiplying matrix P and X_0 (matrix multiplication).
X1 = P @ X0
print(f'Sum of columns of P: {sum(P)}')
print(f'X1:\n{X1}')

#getting the eigen vector and eigen value of transformation matrix p
eig_values,eig_vectors = np.linalg.eig(P)
X_inf = eig_vectors[:,0]
print('Eigen vector associated with eigen value 1',X_inf)
