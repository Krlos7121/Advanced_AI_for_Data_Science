# Medical cost prediction using Bagging (Bootstrap Aggregation)

import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.ensemble import BaggingRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import train_test_split, KFold, cross_val_score
import numpy as np
import pandas as pd
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

"""
1. Data Loading & Preprocessing
"""

df = pd.read_csv('insurance.csv')

# One-hot encoding of categorical variables (replaces string_replacements)
# This creates binary columns for each category (e.g., sex_male, sex_female)
categorical_cols = ['sex', 'smoker', 'region']
df = pd.get_dummies(df, columns=categorical_cols, dtype=int)

# Separate features (X) from target variable (y)
X = df.drop(columns=["charges"])  # Keep: age, bmi, children, etc.
y = df["charges"]  # Medical charges to predict

"""
2. Data Splitting 
Train/test split (75%/25%)
"""

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42
)

"""
3. Baseline Model - Single Decision Tree
"""

# K-Fold cross-validation for baseline decision tree (only train set)
base_tree = DecisionTreeRegressor(
    random_state=42,
    max_depth=10,
    min_samples_leaf=4
)
kfold = KFold(n_splits=7, shuffle=True, random_state=42)
cv_scores_neg = cross_val_score(
    base_tree, X_train, y_train,
    cv=kfold, scoring='neg_mean_squared_error'
)

# Convert negative MSE scores to positive
cv_mse_scores = -cv_scores_neg
print(f"Base Tree CV MSE scores: {cv_mse_scores}")
print(f"Base Tree CV MSE mean: {cv_mse_scores.mean():.2f}")

# Metric reporting function


def report_regression(name, y_true, y_pred):
    mse = mean_squared_error(y_true, y_pred)
    rmse = mse ** 0.5
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    print(f"{name} -> MSE: {mse:.2f} | RMSE: {rmse:.2f} | MAE: {mae:.2f} | R²: {r2:.4f}")
    return mse, rmse, mae, r2


"""
4. Bagging Model
"""

# Baseline Bagging (50 estimators)
bagging = BaggingRegressor(
    estimator=DecisionTreeRegressor(
        random_state=42, max_depth=10, min_samples_leaf=4),
    n_estimators=50,
    oob_score=False,
    random_state=42,
    n_jobs=1
)
# Training the bagging model
bagging.fit(X_train, y_train)

# Predictions on train and test
y_pred_train = bagging.predict(X_train)
y_pred_test = bagging.predict(X_test)

"""
5. Model Evaluation
"""

# Show estimations for each data subset
print("\nBaseline Bagging (50 estimators):")
train_mse_bag, _, _, train_r2_bag = report_regression(
    "Train", y_train, y_pred_train)
test_mse_bag,  _, _, test_r2_bag = report_regression(
    "Test", y_test, y_pred_test)

# Fit base tree for comparison (test set)
base_tree.fit(X_train, y_train)
y_pred_base = base_tree.predict(X_test)
base_mse = mean_squared_error(y_test, y_pred_base)
base_r2 = r2_score(y_test, y_pred_base)

"""
6. Checking the effect of the number of estimators
"""

# Varying n_estimators and recording train/test MSE
n_estimators_list = [5, 10, 15, 20, 30, 50, 100, 150, 200]
train_mse_curve = []
test_mse_curve = []

for n_est in n_estimators_list:
    model = BaggingRegressor(
        estimator=DecisionTreeRegressor(
            random_state=42, max_depth=10, min_samples_leaf=4),
        n_estimators=n_est,
        random_state=42,
        n_jobs=1
    )

    model.fit(X_train, y_train)
    pred_train_curve = model.predict(X_train)
    pred_test_curve = model.predict(X_test)

    train_mse_curve.append(mean_squared_error(y_train, pred_train_curve))
    test_mse_curve.append(mean_squared_error(y_test,  pred_test_curve))

# Baseline bagging predictions for scatter plot
y_pred_final = y_pred_test

"""
7. Visualization
"""
# 2x2 Figure
fig, axes = plt.subplots(2, 2, figsize=(14, 8))

# Subplot 1: Predictions vs Actual (Bagging 50)
axes[0, 0].scatter(y_test, y_pred_final, alpha=0.6, edgecolor='k')
min_val = min(y_test.min(), y_pred_final.min())
max_val = max(y_test.max(), y_pred_final.max())
axes[0, 0].plot([min_val, max_val], [min_val, max_val],
                'r--', label='Ideal line')
axes[0, 0].set_title('Predicted vs Actual (Bagging 50)')
axes[0, 0].set_xlabel('Actual (charges)')
axes[0, 0].set_ylabel('Predicted')
axes[0, 0].legend()

# Subplot 2: Error curve (Train vs Test MSE)
axes[0, 1].plot(n_estimators_list, train_mse_curve,
                marker='o', label='Train MSE')
axes[0, 1].plot(n_estimators_list, test_mse_curve,
                marker='o', label='Test MSE')
axes[0, 1].set_title('Error Curve vs n_estimators')
axes[0, 1].set_xlabel('n_estimators')
axes[0, 1].set_ylabel('MSE')
axes[0, 1].grid(alpha=0.4)
axes[0, 1].legend()

# Subplot 3: MSE comparison
models = ['Base Tree', 'Bagging 50']
mse_vals = [base_mse, test_mse_bag]
bars = axes[1, 0].bar(models, mse_vals,  color=[
    'lightcoral', 'skyblue'])
axes[1, 0].set_title('MSE Comparison (Test)')
axes[1, 0].set_ylabel('MSE')
for b in bars:
    axes[1, 0].text(b.get_x() + b.get_width()/2, b.get_height(),
                    f"{b.get_height():.0f}", ha='center', va='bottom', fontsize=10)

# Subplot 4: R^2 comparison
r2_vals = [base_r2, test_r2_bag]
bars2 = axes[1, 1].bar(models, r2_vals, color=[
    'lightcoral', 'skyblue'])
axes[1, 1].set_title('R² Comparison (Test)')
axes[1, 1].set_ylabel('R²')
for b in bars2:
    axes[1, 1].text(b.get_x() + b.get_width()/2, b.get_height(),
                    f"{b.get_height():.3f}", ha='center', va='bottom', fontsize=10)

plt.tight_layout()
plt.show()

"""
8. Summary / Feature Importance
"""

# Feature importance averaged across bagging estimators
feature_importances = np.mean(
    [est.feature_importances_ for est in bagging.estimators_],
    axis=0
)
fi_series = pd.Series(feature_importances,
                      index=X_train.columns).sort_values(ascending=False)
print("\nTop feature importances:")
print(fi_series)
