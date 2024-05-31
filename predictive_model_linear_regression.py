# File path: predictive_model_linear_regression.py

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
import seaborn as sns

# Step 1: Load the dataset
url = "https://raw.githubusercontent.com/stedy/Machine-Learning-with-R-datasets/master/insurance.csv"
data = pd.read_csv(url)

# Step 2: Explore the data
print(data.head())
print(data.info())
print(data.describe())

# Visualize relationships between features and target variable
sns.pairplot(data)
plt.show()

# Step 3: Preprocess the data
# Handle missing values if any (there are no missing values in this dataset)
# Encode categorical variables
data = pd.get_dummies(data, drop_first=True)

# Separate features and target variable
X = data.drop('charges', axis=1)
y = data['charges']

# Step 4: Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 5: Train the model
model = LinearRegression()
model.fit(X_train, y_train)

# Step 6: Evaluate the model
y_pred = model.predict(X_test)

mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"Mean Squared Error: {mse}")
print(f"R^2 Score: {r2}")

# Plotting the actual vs predicted values
plt.scatter(y_test, y_pred)
plt.xlabel("Actual Values")
plt.ylabel("Predicted Values")
plt.title("Actual vs Predicted Values")
plt.show()

# Step 7: Make predictions
# Use the trained model to make predictions on new data
# new_data = pd.read_csv('new_data.csv')
# predictions = model.predict(new_data)
# print(predictions)
