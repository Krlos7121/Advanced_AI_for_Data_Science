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

# Split the data into train (60%), validation (15%), and test (25%)
df = df.sample(frac=1, random_state=42).reset_index(drop=True)
cut_train = int(len(df) * 0.6)
cut_val = int(len(df) * 0.75)
train_df = df.iloc[:cut_train].copy()
val_df = df.iloc[cut_train:cut_val].copy()
test_df = df.iloc[cut_val:].copy()

# Train/Test separation
X_train = train_df.drop(columns=['charges']).copy()
y_train = train_df['charges'].copy()
X_val = val_df.drop(columns=['charges']).copy()
y_val = val_df['charges'].copy()
X_test = test_df.drop(columns=['charges']).copy()
y_test = test_df['charges'].copy()

# Feature scaling
train_data = X_train.copy()
val_data = X_val.copy()
test_data = X_test.copy()
cols_to_normalize = ['age', 'children']
for col in cols_to_normalize:
    if col in train_data.columns:
        min_val = train_data[col].min()
        max_val = train_data[col].max()
        if max_val != min_val:
            train_data[col] = (train_data[col] - min_val) / (max_val - min_val)
            val_data[col] = (val_data[col] - min_val) / (max_val - min_val)
            test_data[col] = (test_data[col] - min_val) / (max_val - min_val)

# Target (y)
y = y_train.reset_index(drop=True)
train_data = train_data.reset_index(drop=True)
test_data = test_data.reset_index(drop=True)
y_test = y_test.reset_index(drop=True)
val_data = val_data.reset_index(drop=True)
y_val = y_val.reset_index(drop=True)

# Initialize parameters
n_features = train_data.shape[1]
theta = [0.0] * n_features
bias = y.mean()
learning_rate = 0.002
epochs = 10000
error_history = []
train_history = []
val_history = []


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
    # Train cost
    train_cost = mse(train_data, theta, bias, y)
    train_history.append(train_cost)

    # Validation cost
    val_cost = mse(val_data, theta, bias, y_val)
    val_history.append(val_cost)

    if epoch % 100 == 0:
        print(
            f"Epoch {epoch}: Train MSE = {train_cost:.4f}, Val MSE = {val_cost:.4f}")

    if train_cost <= 0.01:
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
    # Get prediction for each instance
    train_predictions.append(hypothesis_theta(
        train_data.iloc[i].tolist(), theta, bias))
train_predictions = pd.Series(train_predictions)

# Train metrics
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

# Test metrics
mse_test = np.mean((test_predictions.values - y_test.values) ** 2)
ss_res = np.sum((y_test.values - test_predictions.values) ** 2)
ss_tot = np.sum((y_test.values - y_test.mean()) ** 2)
r2 = 1 - ss_res / ss_tot

print("Test set MSE:", mse_test)
print("Test set R^2:", r2)
print("Validation set MSE:", val_history[-1])
print("Validation set R^2:", 1 - (val_history[-1] * len(val_data)) /
      np.sum((y_val.values - y_val.mean()) ** 2))

fig, axes = plt.subplots(2, 2, figsize=(12, 8))

corr = df.corr(numeric_only=True)
sns.heatmap(corr, cmap="Greens", annot=True, fmt=".2f", ax=axes[0, 0])
axes[0, 0].set_title("Correlation between features")

axes[0, 1].scatter(range(len(y_test)), y_test,
                   color='blue', label='Real (test)')
axes[0, 1].scatter(range(len(test_predictions)),
                   test_predictions, color='red', label='Predicted (test)')
axes[0, 1].set_title('Real vs Predicted (Test set)')
axes[0, 1].set_xlabel('Index')
axes[0, 1].set_ylabel('Charges')
axes[0, 1].legend()

r2_scores = [r2_train, r2]
set_names = ['Train', 'Test']
axes[1, 1].bar(set_names, r2_scores, color=['skyblue', 'lightcoral'])
axes[1, 1].set_ylim(0, 1)
axes[1, 1].set_title('R^2 Comparison')
axes[1, 1].set_ylabel('R^2 Score')
for i, v in enumerate(r2_scores):
    axes[1, 1].text(i, v + 0.02, f"{v:.2f}", ha='center', fontweight='bold')

axes[1, 0].plot(range(len(train_history)), train_history,
                label='Train MSE', color='skyblue')
axes[1, 0].plot(range(len(val_history)), val_history,
                label='Validation MSE', color='lightcoral')
axes[1, 0].set_title('Train vs Validation Error')
axes[1, 0].legend()

plt.tight_layout()
plt.show()

print("\nDescriptive statistics:\n", df.describe())
