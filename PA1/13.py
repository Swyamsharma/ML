import numpy as np
import random as r
matrix = np.random.randint(1, 11, size=(4, 5), dtype=int)
transpose_matrix = np.transpose(matrix)
print(matrix)
print(transpose_matrix)
arrOfZeros = np.zeros(10)
arrOfOnes = np.ones(10)
arrOfFives = np.ones(10) * 5
arrOfOesZerosFives = np.concatenate((arrOfOnes, arrOfZeros, arrOfFives))
print(arrOfOesZerosFives)
arrOfEvenIntegers = np.arange(10, 51, 2)
print(arrOfEvenIntegers)
random_number = np.random.random()
print(random_number)
np.savetxt('/run/media/vector/New Volume/ML/PA1/matrix.txt', matrix)
loaded_matrix = np.loadtxt('/run/media/vector/New Volume/ML/PA1/matrix.txt')
print(loaded_matrix)
