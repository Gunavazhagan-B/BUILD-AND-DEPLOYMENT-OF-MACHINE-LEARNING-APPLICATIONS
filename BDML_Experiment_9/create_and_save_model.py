# create_and_save_model.py
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
import joblib

print("--- Step 0: Creating and Saving a Dummy ML Model ---")

# 1. Generate a simple dataset
np.random.seed(42)
X = np.random.rand(100, 5) * 10  # 100 samples, 5 features
y = 2 * X[:, 0] + 3 * X[:, 1] - 0.5 * X[:, 2] + 10 + np.random.randn(100) * 2

# Convert to DataFrame (for clarity)
df_model = pd.DataFrame(X, columns=[f'feature_{i+1}' for i in range(X.shape[1])])
df_model['target'] = y

print("\nSample Data Head:")
print(df_model.head())

# 2. Train Linear Regression model
model = LinearRegression()
model.fit(X, y)

# 3. Save model
model_filename = 'linear_regression_model.pkl'
joblib.dump(model, model_filename)

print(f"\nModel saved as: {model_filename}")
print("\nDummy model created and saved successfully. Proceed to create app.py, requirements.txt, and Dockerfile.")
