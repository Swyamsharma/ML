import numpy as np
import pandas as pd
from scipy import stats

dataset = pd.read_csv('/run/media/minion/New Volume/ML/PA1/salary_data.csv')
row_mean = np.mean(dataset, axis=1)
column_mean = np.mean(dataset, axis=0)
overall_mean = np.mean(dataset)
row_median = np.median(dataset, axis=1)
column_median = np.median(dataset, axis=0)
overall_median = np.median(dataset)
row_mode = stats.mode(dataset, axis=1).mode[0]
column_mode = stats.mode(dataset, axis=0).mode[0]
overall_mode = stats.mode(dataset).mode[0]
row_std = np.std(dataset, axis=1)
column_std = np.std(dataset, axis=0)
overall_std = np.std(dataset)


print("Row-wise mean:", row_mean)
print("Column-wise mean:", column_mean)
print("Overall mean:", overall_mean)
print("Row-wise median:", row_median)
print("Column-wise median:", column_median)
print("Overall median:", overall_median)
print("Row-wise mode:", row_mode)
print("Column-wise mode:", column_mode)
print("Overall mode:", overall_mode)
print("Row-wise standard deviation:", row_std)
print("Column-wise standard deviation:", column_std)
print("Overall standard deviation:", overall_std)