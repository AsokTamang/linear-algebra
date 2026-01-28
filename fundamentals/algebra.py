import numpy as np
import matplotlib.pyplot as plt
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
combined_matrix = np.hstack(( coefficients,constants.reshape(2,1)))

#plotting the line from these two system of eqautions
plt.figure()
x=np.linspace(-10,10,100)  #this one is for plotting the line as x-cordinate
x1,y1,z1=combined_matrix[0]
x2,y2,z2= combined_matrix[1]
y1 = (7 + x) / 3  #getting the value of y from 1st equation
y2 = (1 - (3*x)) / 3  #here too but from the second equation
plt.plot(x,y1,color="red")
plt.plot(x,y2,color="blue")
plt.show()


#System of Linear Equations with No Solutions
a = np.array([
    [-1,3],
    [3,-9]
], dtype=np.dtype(float))
b = np.array([7,1],dtype=np.dtype(float))

det_of_a = np.linalg.det(a)
print(det_of_a)
#as the determinant of the matrix consisting of coefficients is 0, so this system of eqaution doesn't has unique solution.It might has infinitely many or no solutions at all.
combined_2 = np.hstack((a,b.reshape(2,1)))
print(combined_2)


plt.figure()
x=np.linspace(-10,10,100)  #this one is for plotting the line as x-cordinate
x11,y11,z11=combined_2[0]
x12,y12,z12= combined_2[1]
y11 = (7 + x) / 3  #getting the value of y from 1st equation
y12 = ((3*x)-1) / 9  #here too but from the second equation
plt.plot(x,y11,color="red")
plt.plot(x,y12,color="blue")
plt.show()
#from the figure, we can observe that the two lines are parallel to each other, so there are no solutions to these two system of eqautions

#as we cannot solve the singular matrix, we will get an error while trying to solve the matrix of coefficients and constants of singular matrix using linalg.solve method
try:
    np.linalg.solve(a, b)  # since the matrix is a singular matrix , we are getting an error
except np.linalg.LinAlgError as err:
    print(err)

third_coef = np.array([[-1,1],[3,-9]])
third_const = np.array([7,-21])
final_matrix = np.hstack((third_coef,third_const.reshape(2,1)))
print(final_matrix)
#system of equation having infinitely many solutions
x= np.linspace(-10,10,100)
x7,y7,z7=final_matrix[0]  #here we are extracting the coefficients of x and y and the constant z from the first equation
x8,y8,z8 = final_matrix[1]
y7 = (7+x) / 3
y8=((3*x)+21) / 9
plt.plot(x,y7)
plt.plot(x,y8)
plt.show()


try:
    np.linalg.solve(third_coef, third_const)  # since the matrix is a singular matrix , we are getting an error
except np.linalg.LinAlgError as err:
    print(err)




#Representing and Solving a System of Linear Equations using Matrices

