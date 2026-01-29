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


#function T_stretch

def T_stretch(a, v):

   
    # Define the transformation matrix
    T = np.array([[2, 0], [0, 1]])

    # Compute the transformation
    w = a * v  #stretching the vector v by a
   

    return w

#function T_rotation
# GRADED FUNCTION: T_rotation
def T_rotation(theta, v):
    # Define the transformation matrix
    T = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])

    # Compute the transformation
    w = T @ v
    return w

#combined 2D rotation and stretching
def T_rotation_and_stretch(theta, a, v):
   rotation_T = np.array([[np.cos(theta),-np.sin(theta)], [np.sin(theta),np.cos(theta)]])  #designing a rotational transformer matrix
   stretch_T = a*v  #stretching the vector v by scalar a
   w = rotation_T @ stretch_T
   return w


#FUNCTION: forward_propagation

def forward_propagation(X, parameters):
    # Retrieving each parameter from the dictionary "parameters".
    W = parameters["W"]
    b = parameters["b"]

    # Implement Forward Propagation to calculate Z.
    ### START CODE HERE ### (~ 2 lines of code)
    Z = (W @ X) + b
    Y_hat = Z


    return Y_hat
print(forward_propagation(0,{'W': np.array([[-0.00607548, -0.00126136]]), 'b': np.array([[0.]])}))