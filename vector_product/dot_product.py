import numpy as np
x = np.array([1,2,3])
y=np.array([4,5,6])
def dot_product(x,y):
    s=0
    for num1,num2 in zip(x, y):
        s+=num1*num2
    return s
print(dot_product(x,y))