# Medical cost prediction (Manual implementation)

import pandas as pd
import numpy as np
import warnings
import matplotlib.pyplot as plt
import seaborn as sns
warnings.simplefilter(action='ignore', category=FutureWarning)

# Load dataset
df = pd.read_csv('insurance.csv')

# One-hot encoding of categorical variables (replaces string_replacements)
categorical_cols = ['sex', 'smoker', 'region']
for col in categorical_cols:
    dummies = pd.get_dummies(df[col], prefix=col)
    df = pd.concat([df.drop(columns=[col]), dummies], axis=1)

# Round all values to 2 decimal places
df = df.round(2)

# Winsorization on charges


def winsorize_columns(df, columns):
    for col in columns:
        lower = df[col].quantile(0.05)
        upper = df[col].quantile(0.95)
        df[col] = np.where(df[col] < lower, lower, df[col])
        df[col] = np.where(df[col] > upper, upper, df[col])
    return df


df = winsorize_columns(df, ['charges'])

# Split the data (70% train, 30% test)
df = df.sample(frac=1, random_state=42).reset_index(drop=True)
cut = int(len(df) * 0.7)
train_df = df.iloc[:cut].copy()
test_df = df.iloc[cut:].copy()

# Train/Test separation
X_train = train_df.drop(columns=['charges']).copy()
y_train = train_df['charges'].copy()
X_test = test_df.drop(columns=['charges']).copy()
y_test = test_df['charges'].copy()

# Feature scaling
train_data = X_train.copy()
test_data = X_test.copy()
cols_to_normalize = ['age', 'children']
for col in cols_to_normalize:
    if col in train_data.columns:
        min_val = train_data[col].min()
        max_val = train_data[col].max()
        if max_val != min_val:
            train_data[col] = (train_data[col] - min_val) / (max_val - min_val)
            test_data[col] = (test_data[col] - min_val) / (max_val - min_val)

# Target (y)
y = y_train.reset_index(drop=True)
train_data = train_data.reset_index(drop=True)
test_data = test_data.reset_index(drop=True)

# Initialize parameters
n_features = train_data.shape[1]
theta = [0.0] * n_features
bias = y.mean()
learning_rate = 0.002
epochs = 10000
error_history = []

# Correlation heatmap (optional diagnostic)
corr = df.corr(numeric_only=True)
plt.figure(figsize=(8, 6))
sns.heatmap(corr, cmap="Greens", annot=True, fmt=".2f")
plt.title("Correlation between features")
plt.tight_layout()
plt.show()

# Hypothesis


def hypothesis_theta(row_values, theta, bias):
    pred = 0.0
    for i in range(len(theta)):
        pred += row_values[i] * theta[i]
    return pred + bias

# Mean Squared Error (cost)


def mse(train_data, theta, bias, y):
    total = 0.0
    rows = len(train_data)
    for i in range(rows):
        pred = hypothesis_theta(train_data.iloc[i].tolist(), theta, bias)
        diff = pred - y.iloc[i]
        total += diff * diff
    return total / (2 * rows)

# Update weights (batch gradient descent)


def update_weights(train_data, theta, bias, y, learning_rate):
    rows = len(train_data)
    features = len(theta)
    theta_updated = list(theta)
    theta_grad = [0.0] * features
    bias_grad = 0.0

    for i in range(rows):
        row_vals = train_data.iloc[i].tolist()
        pred = hypothesis_theta(row_vals, theta, bias)
        err = pred - y.iloc[i]
        for j in range(features):
            theta_grad[j] += err * row_vals[j]
        bias_grad += err

    for j in range(features):
        theta_updated[j] -= (learning_rate / rows) * theta_grad[j]
    bias_updated = bias - (learning_rate / rows) * bias_grad
    return theta_updated, bias_updated


# Training loop
epoch = 0
while epoch < epochs:
    curr_cost = mse(train_data, theta, bias, y)
    error_history.append(curr_cost)
    if curr_cost <= 0.01:
        break
    theta, bias = update_weights(train_data, theta, bias, y, learning_rate)
    epoch += 1

print("Final Theta:", theta)
print("Final Bias:", bias)

# Predictions on train
_ = [hypothesis_theta(train_data.iloc[i].tolist(), theta, bias)
     for i in range(len(train_data))]

train_predictions = []
for i in range(len(train_data)):
    train_predictions.append(hypothesis_theta(
        train_data.iloc[i].tolist(), theta, bias))
train_predictions = pd.Series(train_predictions)

mse_train = np.mean((train_predictions.values - y.values) ** 2)
ss_res_train = np.sum((y.values - train_predictions.values) ** 2)
ss_tot_train = np.sum((y.values - y.mean()) ** 2)
r2_train = 1 - ss_res_train / ss_tot_train
print("Train set MSE:", mse_train)
print("Train set R^2:", r2_train)

# Predictions on test
test_predictions = []
for i in range(len(test_data)):
    test_predictions.append(hypothesis_theta(
        test_data.iloc[i].tolist(), theta, bias))
test_predictions = pd.Series(test_predictions)

# Metrics (original scale)
mse_test = np.mean((test_predictions.values - y_test.values) ** 2)
ss_res = np.sum((y_test.values - test_predictions.values) ** 2)
ss_tot = np.sum((y_test.values - y_test.mean()) ** 2)
r2 = 1 - ss_res / ss_tot

print("Test set MSE:", mse_test)
print("Test set R^2:", r2)
#
# Plot: Real vs Predicted
plt.figure(figsize=(8, 6))
plt.scatter(range(len(y_test)), y_test, color='blue', label='Real (test)')
plt.scatter(range(len(test_predictions)), test_predictions,
            color='red', label='Predicted (test)')
plt.title('Real vs Predicted (Test set)')
plt.xlabel('Index')
plt.ylabel('Charges')
plt.legend()
plt.tight_layout()
plt.show()

# Error convergence
if len(error_history) < epochs:
    error_history += [error_history[-1]] * (epochs - len(error_history))
plt.figure(figsize=(8, 6))
plt.plot(range(epochs), error_history, label='Cost (MSE)')
plt.title('Error Convergence')
plt.xlabel('Epoch')
plt.ylabel('Cost')
plt.legend()
plt.tight_layout()
plt.show()

# Distribution
plt.figure(figsize=(10, 6))
df['charges'].hist(bins=30)
plt.title('Distribution of Charges')
plt.xlabel('Charges')
plt.ylabel('Frequency')
plt.tight_layout()
plt.show()

# Info
print("DataFrame shape:", df.shape)
print("\nDescriptive statistics:\n", df.describe())
print("\nColumn info:")
print(df.info())
print("\nUnique values per column:")
print(df.nunique())
print("\nFirst 5 rows:\n", df.head())
