import numpy as np

arr = [[1,6,7,9],[7,9,3,5]]
print(len(arr), "x", len(arr[0]))
arr = np.array([[1,6,7,9],[7,9,3,5]])

print("i. Dimension of the array:", arr.ndim)
print("ii. Shape of the array:", arr.shape)
print("iii. Size of the array:", arr.size)