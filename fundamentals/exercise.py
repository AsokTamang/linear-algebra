import numpy as np

M = np.array([
[1, 3, 6],
[0, -5, 2],
[-4, 5, 8]
])

def swap_rows(M, row_index_1, row_index_2):  #this function is used for swapping the rows of the matrix
    # Copying matrix M so the changes do not affect the original matrix.
    M = M.copy()
    # Swap indexes
    M[[row_index_1, row_index_2]] = M[[row_index_2, row_index_1]]
    return M



def get_index_first_non_zero_value_from_column(M,starting_row,column):
  for i,val in enumerate(M[starting_row:,column]):  #here this for loop loops in a specific array obtained from the starting row and a column
   if not np.isclose(val,0,atol=1e-08):  #if we find the fist non_zero number in this given row and column index then
      index = i + starting_row
      return index
  return -1  #if we didn't find any non_zero value then we just return -1
print(get_index_first_non_zero_value_from_column(1,1))


def find_first_non_zerovalue_rowwise(M,row,augmented=False):
    M = M.copy()
    if augmented==True:
        M=M[:,:-1]  #excluding the constant values
    row_array = M[row]
    for i,val in enumerate(row_array):
        if not np.isclose(val,0,atol=1e-08):  #if we found the very first non_zero element then we just return the row index
            return i
    return -1
print(find_first_non_zerovalue_rowwise(0,M,augmented=False))


#calculating the row_echelon form and reduced row_echelon form
# GRADED FUNCTION: row_echelon_form

def augmented_matrix(A, B):
    return np.hstack((A,B))



def row_echelon_form(A, B):


    # Before any computation, checking if matrix A (coefficient matrix) has non-zero determinant.
    # It will use the numpy sub library np.linalg to compute it.

    det_A = np.linalg.det(A)

    # Returning "Singular system" if determinant is zero  as we cannot compute the row echelon form for singular system
    if np.isclose(det_A, 0) == True:
        return 'Singular system'

    # Making copies of the input matrices to avoid modifying the originals
    A = A.copy()
    B = B.copy()

    # Converting matrices to float to prevent integer division
    A = A.astype('float64')
    B = B.astype('float64')

    # Number of rows in the coefficient matrix
    num_rows = len(A)

    #LOGIC
    # Transforming matrices A and B into the augmented matrix M
    M = augmented_matrix(A, B)

    # Iterate over the rows.
    for row in range(num_rows):
        #the pivot candidates or the elements at the main diagonal have same number of rows and columns
        pivot_candidate = M[row, row]
        #checking if the current pivot candidate is zero or not
        if np.isclose(pivot_candidate, 0) == True:
            # Get the index of the first non-zero value below the pivot_candidate.
            first_non_zero_value_below_pivot_candidate = get_index_first_non_zero_value_from_column(M, row, row)

            #swapping the rows as the current pivot candidate was 0
            M = swap_rows(M, row, first_non_zero_value_below_pivot_candidate)

            # Get the pivot, which is in the main diagonal now
            pivot = M[row, row]   #now we get the non_Zero element at current pivot position

            # If pivot_candidate is already non-zero, then it is the pivot for this row
        else:
            pivot = pivot_candidate

            # Now applying the row reduction in every row below the current

        # Dividing the current row by the pivot, so the current pivot will be 1 to produce row reduced echolon form
        # Where current_row can be accessed using M[row].
        M[row] = (1 / pivot) * M[row]

        # Performing row reduction for rows below the current row
        for j in range(row + 1, num_rows):
            value_below_pivot = M[j, row]  # as the column is same, and the column index in this case is row  #same column but different row

            # Performing row reduction using the formula:
            #as the pivot in the above row is already 1. so we are using formula like this m[row]=m[row] - pivot * m[row]
            M[j] = M[j] - value_below_pivot * M[row]
    return M  #now the first column is done



