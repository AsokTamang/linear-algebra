import numpy as np

A = np.array([[4, 9, 9], [9, 1, 6], [9, 2, 3]])
print("Matrix A (3 by 3):\n", A)

B = np.array([[2, 2], [5, 7], [4, 4]])
print("Matrix B (3 by 2):\n", B)

print(np.matmul(A, B))  #using the api matmul
#inorder for the matrix multiplication between two matrices A and B to be possible, the columns of the matrix A or first matrix must be equal to the row of matrix B or second matrix.
print(A.ndim)  #here as our array A has rows and columns, so it is a two-dimensional array

print(np.zeros((3,1)))


