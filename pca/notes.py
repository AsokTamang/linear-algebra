#so the principal component analysis(PCA) is the technique which is used for converting the large datasets into smaller dimensional dataset without loosing the
#majority of datas,
#REVERSE ENGINEERING THOUGHT PROCESS

#we need the eigen vector to multiply eigen vector over the norm of eigen vector with the covariance matrix of the original dataset

#and the covariance matrix can be obtained using the transpose of X - MEAN(X)*X - MEAN(X) / (number of features - 1)  or the other way is using the  matrix of [varx cov(x,y) cov(y,x) vary]
#then using this covariance matrix we can obtain the eigen value using the characteristic polynomial
#which can be done using det(cov_matrix - eigenvalue*identity_matrix of same size)
#then choosing higher eigen value, we get eigen vector by preserving the original dataset info
#then we project the original data set using cov_matrix of original dataset * eigen_vector / (norm of eigen vector)

import numpy as np
import matplotlib.pyplot as plt

A=np.array([[2,3],[3,1]])


A_eig = np.linalg.eig(A)
eigen_value=A_eig[0]
eigen_vector = A_eig[1]
print(eigen_value)
print(eigen_vector)