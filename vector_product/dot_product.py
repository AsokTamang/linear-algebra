import numpy as np
x = np.array([1,2,3])
y=np.array([4,5,6])
def dot_product(x,y):
    s=0
    for num1,num2 in zip(x, y):
        s+=num1*num2
    return s
print(dot_product(x,y))




#while doing the dot products, its preferred to do with the vectorized form rather than the loop form like above
#the vectorized form is:
print(x@y)   #we can also use @ inorder to dot product the vectors x and y
#as the vectorized form of dot product is much more faster than the loop form



