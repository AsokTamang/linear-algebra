import numpy as np


def T(v):
    w = np.zeros((3, 1))
    w[0, 0] = 3 * v[0, 0]
    w[2, 0] = -2 * v[1, 0]

    return w


v = np.array([[3], [5]])
w = T(v)

print("Original vector:\n", v, "\n\n Result of the transformation using the function T is:\n", w)


#hotizontal scaling
def T_hscaling(v):
    A = np.array([[2, 0], [0, 1]])
    w = A @ v

    return w


def transform_vectors(T, v1, v2):
    V = np.hstack((v1, v2))
    W = T(V)

    return W


e1 = np.array([[1], [0]])
e2 = np.array([[0], [1]])

transformation_result_hscaling = transform_vectors(T_hscaling, e1, e2)

print("Original vectors:\n e1= \n", e1, "\n e2=\n", e2,"\n\n Result of the transformation (matrix form):\n", transformation_result_hscaling)


print('The horizontal stack is:',np.hstack((e1,e2)))