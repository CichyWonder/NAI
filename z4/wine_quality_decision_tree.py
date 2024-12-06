import numpy as np
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor

# Load the data
input_file = 'winequality-white.txt'
data = np.loadtxt(input_file, delimiter=';')
X, y = data[:, :-1], data[:, -1]

# Split data into training and testing datasets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=5)

# Instantiate and train Decision Tree Regressor
regressor = DecisionTreeRegressor(random_state=0, max_depth=8)
regressor.fit(X_train, y_train)

# Make predictions on the test set
y_test_pred = regressor.predict(X_test)

# Evaluate performance
rmse_train = np.sqrt(mean_squared_error(y_train, regressor.predict(X_train)))
rmse_test = np.sqrt(mean_squared_error(y_test, y_test_pred))
print("\nRegressor performance on training dataset\n")
print(f"RMSE: {rmse_train:.4f}")

print("\nRegressor performance on test dataset\n")
print(f"RMSE: {rmse_test:.4f}")
