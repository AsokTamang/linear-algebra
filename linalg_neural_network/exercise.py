import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


def T(v):
    w = np.zeros((3, 1))
    w[0, 0] = 3 * v[0, 0]
    w[2, 0] = -2 * v[1, 0]

    return w


v = np.array([[3], [5]])
w = T(v)

print("Original vector:\n", v, "\n\n Result of the transformation:\n", w)

#stretching a vector v by scalar A means just doing a scalar product between the vector v and scalar A


# GRADED FUNCTION: T_stretch

def T_stretch(a, v):

    ### START CODE HERE ###
    # Define the transformation matrix
    T = np.array([[2, 0], [0, 1]])

    # Compute the transformation
    w = a * v  #stretching the vector v by a
    ### END CODE HERE ###

    return w

