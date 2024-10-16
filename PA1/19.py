import pandas as pd

data = pd.read_csv('/run/media/minion/New Volume/ML/PA1/salary_data.csv')

num_features = data.shape[1] - 1                                                         
num_patterns = data.shape[0]
print(f"Number of features: {num_features}")
print(f"Number of patterns: {num_patterns}")

output_range = (data.iloc[:, -1].min(), data.iloc[:, -1].max())
print(f"Range of output: {output_range}")

def split_dataset(data, train_ratio):
    train_size = int(len(data) * train_ratio)
    train_data = data[:train_size]
    test_data = data[train_size:]
    return train_data, test_data

ratios = [(i / 100, (100 - i) / 100) for i in range(10, 100, 10)]
for train_ratio, test_ratio in ratios:
    train_data, test_data = split_dataset(data, train_ratio)
    print(f"Train size for {int(train_ratio * 100)}:{int(test_ratio * 100)} split: {len(train_data)}")
    print(f"Test size for {int(train_ratio * 100)}:{int(test_ratio * 100)} split: {len(test_data)}")