import numpy as np
matrix = np.random.randint(1, 100, size=(10, 5), dtype=int)
for i in range(10):
    print(np.max(matrix[i]))
    print(np.min(matrix[i]))