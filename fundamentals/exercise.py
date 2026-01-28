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



def find_first_non_zerovalue_columnwise(starting_row,column):
  for i,val in enumerate(M[starting_row:,column]):  #here this for loop loops in a specific array obtained from the starting row and a column
   if not np.isclose(val,0,atol=1e-08):  #if we find the fist non_zero number in this given row and column index then
      index = i + starting_row
      return index
  return -1  #if we didn't find any non_zero value then we just return -1
print(find_first_non_zerovalue_columnwise(1,1))


def find_first_non_zerovalue_rowwise(row,M,augmented=False):
    M = M.copy()
    if augmented==True:
        M=M[:,:-1]  #excluding the constant values
    row_array = M[row]
    for i,val in enumerate(row_array):
        if not np.isclose(val,0,atol=1e-08):  #if we found the very first non_zero element then we just return the row index
            return i
    return -1
print(find_first_non_zerovalue_rowwise(0,M,augmented=False))



