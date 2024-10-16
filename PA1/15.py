import numpy as np
matrix = np.random.randint(1, 100, size=(10, 5), dtype=int)
print(matrix)
num_features = matrix.shape[1]
pattern_count = {}
for i in range(num_features):
    feature_column = matrix[:, i]
    count = np.unique(feature_column, return_counts=True)[1]
    count1 = 0
    for j in count: 
        if j > 1: 
            count1 += j
    print(f'feature {i+1}: {count1}')