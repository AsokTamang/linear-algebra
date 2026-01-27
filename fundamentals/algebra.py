import numpy as np
a = np.array([1,2,3,4])
final = np.reshape(a,(2,2))
print(f'The reshaped array is:', final)
print(f'The shape of final array is:', final.shape)
print(f'The number of elements in a final array is:', final.size)
print('The dimension of final array is', final.ndim)



#mathematical operations
arr_1 = np.array([2, 4, 6])
arr_2 = np.array([1, 3, 5])
print(arr_1 + arr_2)
print(arr_1 - arr_2)
print(arr_1 * arr_2)
two_dim = np.array(([1, 2, 3],
          [4, 5, 6],
          [7, 8, 9]))


print(two_dim[:,2])  #here we are printing only the last column elements from each rows


coefficients = np.array([[-1,3],[3,2]],dtype=np.dtype(float))
constants = np.array([7,1],dtype=np.dtype(float))
print(coefficients.shape)
print(constants)

#calculating the determinant of matrix coefficient
print(np.linalg.det(coefficients))

#horizontally stacking the two matrices
#horizontally stacking the coefficients and constants and as the matrix constants has only one row , so we reshaped it into having 2 rows and 1 column
np.hstack(( coefficients,constants.reshape(2,1)))