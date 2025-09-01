import pandas as pd
import numpy as np
import warnings
import matplotlib.pyplot as plt

# Ignore FutureWarnings
warnings.simplefilter(action='ignore', category=FutureWarning)

# Load the dataset
df = pd.read_csv('insurance.csv')

# BMI classifications


def classify_bmi(bmi):
    if bmi < 18.5:
        return 'underweight'
    elif bmi <= 24.9:
        return 'normal weight'
    elif bmi <= 29.9:
        return 'overweight'
    elif bmi <= 34.9:
        return 'class 1 obesity'
    elif bmi <= 39.9:
        return 'class 2 obesity'
    else:
        return 'class 3 obesity'


df['bmi_category'] = df['bmi'].apply(classify_bmi)

# Mapping string values to numeric values
string_replacements = {
    'sex': {'male': 1, 'female': 2},
    'bmi_category': {'underweight': 1, 'normal weight': 2, 'overweight': 3, 'class 1 obesity': 4, 'class 2 obesity': 5, 'class 3 obesity': 6},
    'smoker': {'yes': 1, 'no': 2},
    'region': {'northeast': 1, 'northwest': 2, 'southeast': 3, 'southwest': 4}
}

df.replace(string_replacements, inplace=True)

# Drop bmi column, I'll be using the newly created bmi_category instead
df.drop(columns=['bmi'], inplace=True)

# Round all values to 2 decimal places
df = df.round(2)

# Applying winsorization to remove outliers at the top and bottom 5%


def winsorize_columns(df, columns):
    for col in columns:
        lower_bound = df[col].quantile(0.05)
        upper_bound = df[col].quantile(0.95)
        df[col] = np.where(df[col] < lower_bound, lower_bound, df[col])
        df[col] = np.where(df[col] > upper_bound, upper_bound, df[col])
    return df


cols_to_winsorize = ['charges']
df = winsorize_columns(df, cols_to_winsorize)

# Divide into test and train DFs
# print("Total rows according to .length: ", len(df)) Result: df was split correctly
train_amount = int(len(df) * 0.7)
train_df = df[:train_amount]
test_df = df[train_amount:]

# Gradient descent parameters
train_data = train_df.drop(columns=["charges"])  # Dependant variables only
y = train_df["charges"]  # Values to predict

test_data = test_df.drop(columns=["charges"])
y_test = test_df["charges"]

# Normalize data before processing, using Min-Max scaling
cols_to_normalize = ['age', 'children']

for col in cols_to_normalize:
    min_val = train_data[col].min()
    max_val = train_data[col].max()
    train_data[col] = (train_data[col] - min_val) / (max_val - min_val)
    test_data[col] = (test_data[col] - min_val) / (max_val - min_val)

# Normalizing y
y_min = y.min()
y_max = y.max()
y = (y - y_min) / (y_max - y_min)
y_test = (y_test - y_min) / (y_max - y_min)

theta = np.zeros(train_data.shape[1])
bias = y.mean()  # Starting bias is the mean of my train set
learning_rate = 0.03
epochs = 3000
gradient = np.zeros(train_data.shape[1])
error = []

# Here is where the GD training happens
# Theta hypothesis


def hypothesis_theta(train_data, theta, bias):
    prediction = 0
    # print("train_data tolist: ", train_data.tolist())
    for i in range(len(train_data)):
        prediction += train_data[i] * theta[i]
    prediction += bias
    return prediction

# Mean Squared Error


def mse(train_data, theta, bias, y):
    cost = 0
    rows = len(train_data)
    for i in range(rows):
        cost += (hypothesis_theta(
            train_data.iloc[i].tolist(), theta, bias) - y[i]) ** 2
    cost /= (2 * rows)
    return cost

# Get new values for our linear regressions independent variables


def update_weights(train_data, theta, bias, y, learning_rate):
    rows = len(train_data)  # m
    features = len(theta)  # n
    theta_updated = list(theta)

    theta_gradient = [0.0] * features
    bias_gradient = 0

    for i in range(rows):
        error = hypothesis_theta(
            train_data.iloc[i].tolist(), theta, bias) - y.iloc[i]
        for j in range(features):
            theta_gradient[j] += error * train_data.iloc[i][j]
        bias_gradient += error

    for j in range(features):
        theta_updated[j] -= (learning_rate / rows) * theta_gradient[j]
    bias_updated = bias - (learning_rate / rows) * bias_gradient

    return theta_updated, bias_updated


# Gradient Descent Loop
i = 0
while i < epochs:
    curr_error = mse(train_data, theta, bias, y)
    error.append(curr_error)
    if curr_error <= 0.01:
        break
    theta, bias = update_weights(train_data, theta, bias, y, learning_rate)
    i += 1

print("Final Theta:", theta)
print("Final Bias:", bias)

result_df = pd.DataFrame([hypothesis_theta(
    train_data.iloc[i].tolist(), theta, bias) for i in range(len(train_data))])
plt.show()

# Predictions with test_data
test_predictions = []
for i in range(len(test_data)):
    test_predictions.append(hypothesis_theta(
        test_data.iloc[i].tolist(), theta, bias))
test_predictions = pd.DataFrame(test_predictions)

# Denormalize test predictions and real values
test_predictions_denormalized = test_predictions * (y_max - y_min) + y_min
y_test_denormalized = y_test * (y_max - y_min) + y_min

# MSE on the test set
mse_test = sum((test_predictions_denormalized.values.flatten(
) - y_test_denormalized.values.flatten()) ** 2) / len(test_predictions_denormalized)
print("Test set MSE:", mse_test)

# Get R^2 in the test set
ss_res = sum((y_test_denormalized.values.flatten() -
             test_predictions_denormalized.values.flatten()) ** 2)
ss_tot = sum((y_test_denormalized.values.flatten() -
             y_test_denormalized.mean()) ** 2)
r2 = 1 - (ss_res / ss_tot)
print("Test set R^2:", r2)

# Predictions vs Real values (test set)
plt.figure(figsize=(8, 6))
plt.scatter(range(len(y_test_denormalized)), y_test_denormalized,
            color='blue', label='Real values (test)')
plt.scatter(range(len(test_predictions_denormalized)),
            test_predictions_denormalized, color='red', label='Predictions (test)')
plt.title('Real values vs Predictions (Test set)')
plt.xlabel('Index')
plt.ylabel('Charges')
plt.legend()
plt.show()

# Error convergence plot
if len(error) < epochs:
    error += [error[-1]] * (epochs - len(error))

plt.figure(figsize=(8, 6))
plt.plot(range(epochs), error, label='Cost Function (MSE)')
plt.title('Error Convergence')
plt.xlabel('Epoch Number')
plt.ylabel('Error')
plt.legend()
plt.show()

# Data distributions
plt.figure(figsize=(10, 6))
df['charges'].hist(bins=30)
plt.title('Distribution of Charges')
plt.xlabel('Charges')
plt.ylabel('Frequency')
plt.show()

# DataFrame information
print("DataFrame shape:", df.shape)
print("\nDescriptive statistics:\n", df.describe())
print("\nColumn info:")
print(df.info())
print("\nNumber of unique values per column:")
print(df.nunique())
print("\nFirst 5 rows:\n", df.head())
