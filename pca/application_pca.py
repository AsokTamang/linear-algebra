import numpy as np
import scipy.sparse.linalg
import utils
import matplotlib.pyplot as plt

imgs = utils.load_images('../data/')
print(imgs[0].reshape(-1))  #flattening the image in one array
height, width = imgs[0].shape

print(f'\nour dataset has {len(imgs)} images of size {height}x{width} pixels\n')
plt.imshow(imgs[0], cmap='gray')
plt.show()

flatten_images = np.array([img.reshape(-1) for img in imgs])  #flattening the image's pixels into single array,row-wise by default


#centering the data of matrix
# Graded cell
def center_data(Y):

    mean_vector = np.mean(Y, axis=0)  # axis=0 means finding the mean column wise, as the Y consists the number of observations in a row and the number of pixels in a column
    mean_matrix = Y - mean_vector  #centering the data, which means subtracting the mean from the variable column wise
    # use np.reshape to reshape into a matrix with the same size as Y. Remember to use order='F'
    mean_matrix = np.reshape(mean_matrix,Y.shape,order='F')

    X = mean_matrix
    return X

X = center_data(flatten_images)

#getting the covariance matrix
def get_cov_matrix(X):
   #as the formulae for getting the covariance matrix is (A.T @ A )/ (no. of observations - 1)
    cov_matrix = np.transpose(X) @ X
    cov_matrix = cov_matrix / ((X.shape[0]) - 1)  # dividing by the number of observations in a matrix which is a row here

    return cov_matrix

cov_matrix = get_cov_matrix(X)

eigenvals, eigenvecs = scipy.sparse.linalg.eigsh(cov_matrix, k=55)
print('Eigen values are:', eigenvals)
print('Eigen vectors are:', eigenvecs)

#sorting the eigen values and eigen vectors from largest to smallest
eigenvals = eigenvals[::-1]
eigenvecs =  eigenvecs[:,::-1]
print('Eigen values are:', eigenvals)
print('Eigen vectors are:', eigenvecs)

def perform_PCA(X, eigenvecs, k):
    V = eigenvecs[:, :k]  #choosing the eigen vectors , k PCAs means largest and second largest eigen values
    Xred = X @ V  #centered dataset dot product eigen vector(K PCAs)
    ### END CODE HERE ###
    return Xred


