"""
==========================================
Teaching a neural network to classify data from winequality-white.txt dataset
Creator:
Michał Cichowski s20695
==========================================
To run program install:
pip install numpy
pip install matplotlib
pip install sklearn
pip install tensorflow
==========================================
Usage:
python neural_network_winequality.py
==========================================
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense, Dropout


# Load the dataset
df = pd.read_csv('winequality-white.txt', sep=',', header=None)

# Split into input (X) and output (y)
X_wine, y_wine = df.values[:, :-1], df.values[:, -1]

# Ensure that all data are floating point values
X_wine = X_wine.astype('float32')

# Encode the labels to one-hot encoding for multi-class classification
y_wine = OneHotEncoder(sparse=False).fit_transform(y_wine.reshape(-1, 1))

# Split into training and test datasets
X_train, X_test, y_train, y_test = train_test_split(X_wine, y_wine, test_size=0.33, random_state=42)

# Determine the number of input features
wine_features = X_wine.shape[1]
num_classes = y_wine.shape[1]

# Model
model = Sequential()
model.add(Dense(64, activation='relu', kernel_initializer='he_normal', input_shape=(wine_features,)))
model.add(Dropout(0.4))
model.add(Dense(32, activation='relu', kernel_initializer='he_normal'))
model.add(Dropout(0.4))
model.add(Dense(num_classes, activation='softmax'))

# Compile model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Train the model
history = model.fit(X_train, y_train, epochs=100, batch_size=32, verbose=1, validation_data=(X_test, y_test))

# Make predictions
pred_test = model.predict(X_test)
pred_test_classes = np.argmax(pred_test, axis=1)
y_test_classes = np.argmax(y_test, axis=1)

# Calculate accuracy score
score = accuracy_score(y_test_classes, pred_test_classes)

# Print summary statistics
print('Accuracy: %.3f' % score)

# Plot learning curves
plt.title('Learning Curves')
plt.xlabel('Epoch')
plt.ylabel('Categorical Cross-Entropy Loss')
plt.plot(history.history['loss'], label='train')
plt.plot(history.history['val_loss'], label='validation')
plt.legend()
plt.show()
