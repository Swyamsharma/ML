import pandas as pd
import numpy as np
num_houses = 10

sizes = np.random.randint(500, 2000, num_houses) 
bedrooms = np.random.randint(1, 6, num_houses)  
ages = np.random.randint(0, 100, num_houses)      
distances = np.random.randint(1, 30, num_houses) 
prices = sizes*10+bedrooms*100+ages*(-10)+distances*(-100)+np.random.randint(-100, 1000, num_houses)

data = {
    'Size': sizes,
    'Bedrooms': bedrooms,
    'Age': ages,
    'Distance': distances,
    'Price': prices
}
df = pd.DataFrame(data)
df.to_csv('house_prices.csv', index=False)
dataset = pd.read_csv('house_prices.csv')
print(dataset)

dataset['Size'] = dataset['Size']/dataset['Size'].max()
dataset['Bedrooms'] = dataset['Bedrooms']/dataset['Bedrooms'].max()
dataset['Age'] = dataset['Age']/dataset['Age'].max()
dataset['Distance'] = dataset['Distance']/dataset['Distance'].max()
price_max = dataset['Price'].max()
dataset['Price'] = dataset['Price']/price_max
X = dataset[['Size', 'Bedrooms', 'Age', 'Distance']].values
y = dataset['Price'].values
X = np.c_[np.ones(X.shape[0]), X]
weights = np.random.randn(X.shape[1])
learning_rate = 0.01
num_iterations = 1000
def gradient_descent(X, y, weights, learning_rate=0.01, num_iterations=1000):
    n = len(y)
    for _ in range(num_iterations):
        y_pred = np.dot(X, weights)
        gradient = np.dot(X.T, y_pred - y) / n
        weights = weights - learning_rate * gradient
    return weights

def mean_squared_error(X, y, weights):
    y_pred = np.dot(X, weights)
    return np.mean((y - y_pred) ** 2)
train_x, train_y = X[:6], y[:6]
val_x, val_y = X[6:8], y[6:8]
test_x, test_y = X[8:], y[8:]
best_val_mse = float('inf')
best_alpha = None
for alphas in [0.01, 0.1, 0.5, 1]:
    weight1 = gradient_descent(train_x, train_y, weights, learning_rate=alphas)
    val_mse = mean_squared_error(val_x, val_y, weight1)
    if val_mse < best_val_mse:
        best_val_mse = val_mse
        best_alpha = alphas
weights = gradient_descent(train_x, train_y, weights, learning_rate=best_alpha)
print(f'Weights : {weights}')
print(f'MSE : {mean_squared_error(test_x, test_y, weights)}')
test = [1, 2500, 4, 10, 5]
print(f'Price : {np.dot(test, weights)}')