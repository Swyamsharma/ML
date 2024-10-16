import pandas as pd
data = pd.read_csv('/run/media/vector/New Volume/ML/PA1/output.csv')
data = data.iloc[:-1, :-1]
print(data)