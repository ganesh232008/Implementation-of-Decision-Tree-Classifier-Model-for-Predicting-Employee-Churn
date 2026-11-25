# Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn

## AIM:
To write a program to implement the Decision Tree Classifier Model for Predicting Employee Churn.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Program:

```
/*
Program to implement the Decision Tree Classifier Model for Predicting Employee Churn.
Developed by: L Ganesh Kanna 
RegisterNumber: 212225230072 
*/

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn import metrics
import warnings
warnings.filterwarnings("ignore")

# 1) Load dataset
# Replace the path below with the correct path for your environment if needed.
csv_path = "Employee.csv"   # <-- change if file is somewhere else
try:
    data = pd.read_csv(csv_path)
except FileNotFoundError:
    raise FileNotFoundError(f"File not found at {csv_path}. Update 'csv_path' to the correct location.")

# 2) Quick inspection
print("Data shape:", data.shape)
display(data.head())
print("\nInfo:")
display(data.info())
print("\nMissing values per column:\n", data.isnull().sum())
print("\nTarget distribution (left):")
display(data["left"].value_counts())

# 3) Handle missing values (simple approach: drop rows with NA)
if data.isnull().any().any():
    print("\nDropping rows with missing values...")
    data = data.dropna()
    print("New shape after dropna:", data.shape)

# 4) Encode categorical columns
# We encode 'salary'. If you have other categorical columns (e.g., 'department'), consider encoding or one-hot.
if 'salary' in data.columns:
    le = LabelEncoder()
    data['salary'] = le.fit_transform(data['salary'].astype(str))
    print("\nSalary classes (label encoding mapping):")
    mapping = dict(zip(le.classes_, le.transform(le.classes_)))
    print(mapping)

# 5) Define features (X) and target (y)
feature_cols = [
    "satisfaction_level", "last_evaluation", "number_project",
    "average_montly_hours", "time_spend_company",
    "Work_accident", "promotion_last_5years", "salary"
]

# Validate that all feature columns exist
missing_feats = [c for c in feature_cols if c not in data.columns]
if missing_feats:
    raise ValueError(f"Missing expected feature columns in dataset: {missing_feats}")

X = data[feature_cols]
y = data["left"]

display(X.head())

# 6) Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=100, stratify=y if len(y.unique())>1 else None
)
print(f"\nTrain shape: {X_train.shape}, Test shape: {X_test.shape}")

# 7) Initialize and train Decision Tree
dt = DecisionTreeClassifier(criterion="entropy", random_state=42)
dt.fit(X_train, y_train)

# 8) Predict on test data
y_pred = dt.predict(X_test)

# 9) Evaluation
acc = metrics.accuracy_score(y_test, y_pred)
print(f"\nAccuracy on test set: {acc:.4f}")

print("\nConfusion Matrix:")
cm = metrics.confusion_matrix(y_test, y_pred)
print(cm)

print("\nClassification Report:")
print(metrics.classification_report(y_test, y_pred, digits=4))

# 10) Example: predict for a new employee
# Format: [satisfaction_level, last_evaluation, number_project, average_montly_hours, time_spend_company, Work_accident, promotion_last_5years, salary]
# NOTE: salary must use the same encoding as used above (see mapping printed earlier)
new_employee = [[0.5, 0.8, 9, 260, 6, 0, 1, 2]]  # example values
pred = dt.predict(new_employee)
print(f"\nPrediction for new employee {new_employee[0]}: {pred} ->", "left" if pred[0]==1 else "stayed")

# 11) Visualize the decision tree
plt.figure(figsize=(14,10))
plot_tree(
    dt,
    feature_names=feature_cols,
    class_names=['stayed','left'],
    filled=True,
    rounded=True,
    fontsize=10
)
plt.title("Decision Tree for Employee Churn")
plt.show()

# Optionally save the tree figure
# plt.savefig("employee_churn_tree.png", bbox_inches="tight", dpi=200)

# 12) (Optional) show feature importances
importances = pd.Series(dt.feature_importances_, index=feature_cols).sort_values(ascending=False)
print("\nFeature importances:")
display(importances)

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor, plot_tree
from sklearn import metrics
import warnings
warnings.filterwarnings("ignore")

# 1) Load dataset
csv_path = "Salary.csv"    # <-- Change path if needed
try:
    data = pd.read_csv(csv_path)
except FileNotFoundError:
    raise FileNotFoundError(f"File not found at: {csv_path}. Update the path.")

print("Dataset Loaded Successfully!\n")

# 2) Data exploration
print("Shape:", data.shape)
display(data.head())
print("\nInfo:")
display(data.info())
print("\nMissing Values:\n", data.isnull().sum())

# 3) Encode categorical column 'Position'
if "Position" in data.columns:
    le = LabelEncoder()
    data["Position"] = le.fit_transform(data["Position"])
    print("\nLabel Encoding Mapping (Position):")
    mapping = dict(zip(le.classes_, le.transform(le.classes_)))
    print(mapping)

# 4) Select features and target
X = data[["Position", "Level"]]
y = data["Salary"]

print("\nFeature Sample:")
display(X.head())
print("\nTarget Sample:")
display(y.head())

# 5) Train-test split
X_train, X_test, Y_train, Y_test = train_test_split(
    X, y, test_size=0.2, random_state=2
)
print(f"\nTrain Size: {X_train.shape}, Test Size: {X_test.shape}")

# 6) Initialize and train Decision Tree Regressor
dt = DecisionTreeRegressor(random_state=10)
dt.fit(X_train, Y_train)
print("\nModel Training Completed!")

# 7) Predict on test data
y_pred = dt.predict(X_test)
print("\nPredicted Salaries:", y_pred)

# 8) Evaluate model using R² Score
r2 = metrics.r2_score(Y_test, y_pred)
print(f"\nR² Score: {r2:.4f}")

# 9) Visualize Decision Tree
plt.figure(figsize=(12,8))
plot_tree(dt, feature_names=["Position", "Level"], filled=True)
plt.title("Decision Tree Regressor for Salary Prediction")
plt.show()

# 10) Feature Importances
importances = pd.Series(dt.feature_importances_, index=["Position", "Level"])
print("\nFeature Importances:")
display(importances)

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

# Load dataset
df = pd.read_csv('weather-station-eee-block_2024_07_13.csv')

# Display first five rows
print("First Five Rows of Dataset")
print(df.head())

# Convert time column into datetime format
df['time'] = pd.to_datetime(df['time'])

# Extract date and time features
df['month'] = df['time'].dt.month
df['day'] = df['time'].dt.day
df['hour'] = df['time'].dt.hour

# Handle missing values
numeric_columns = df.select_dtypes(include=np.number).columns

for col in numeric_columns:
    df[col].fillna(df[col].mean(), inplace=True)

# Select input features
X = df[['hum', 'pressure', 'wind_speed', 'month', 'day', 'hour']]

# Select target variables
Y = df[['tem', 'pm2_5', 'tsr']]

# Split dataset into training and testing data
X_train, X_test, Y_train, Y_test = train_test_split(
    X,
    Y,
    test_size=0.2,
    random_state=42
)

# Create Random Forest Regression model
model = RandomForestRegressor(
    n_estimators=100,
    random_state=42
)

# Train the model
model.fit(X_train, Y_train)

# Predict output values
Y_pred = model.predict(X_test)

# Evaluate the model
r2 = r2_score(Y_test, Y_pred)
mae = mean_absolute_error(Y_test, Y_pred)
rmse = np.sqrt(mean_squared_error(Y_test, Y_pred))

# Display evaluation metrics
print("\nModel Evaluation Metrics")
print("R2 Score :", r2)
print("MAE :", mae)
print("RMSE :", rmse)

# Perform cross-validation
cv_scores = cross_val_score(
    model,
    X,
    Y,
    cv=5,
    scoring='r2'
)

# Display cross-validation results
print("\nCross Validation Scores")
print(cv_scores)

print("\nAverage Cross Validation Score")
print(cv_scores.mean())

```

## Output:

<img width="1365" height="407" alt="image" src="https://github.com/user-attachments/assets/15965db7-600f-4f0b-a4d6-777b7ed9480e" />

<img width="1350" height="407" alt="image" src="https://github.com/user-attachments/assets/62907b19-c18b-4b28-8ec6-5d72830ec730" />

<img width="1352" height="397" alt="image" src="https://github.com/user-attachments/assets/c0789836-94cd-49e0-a3f5-341f47d8d8c2" />

<img width="1353" height="402" alt="image" src="https://github.com/user-attachments/assets/d1656fda-7ca3-40bd-8058-c803273b0747" />

<img width="1355" height="295" alt="image" src="https://github.com/user-attachments/assets/2affda42-be51-4b5a-9551-86d9dcaf0ed6" />

<img width="1127" height="712" alt="image" src="https://github.com/user-attachments/assets/3f071b78-99b1-472c-910f-383161a74d1c" />

<img width="1363" height="270" alt="image" src="https://github.com/user-attachments/assets/f3ffefaa-ff9b-45bc-9f6f-a4d7b7ebdb0f" />

<img width="1186" height="817" alt="image" src="https://github.com/user-attachments/assets/317fc151-1a39-4519-87d8-7050afea9819" />

<img width="957" height="503" alt="image" src="https://github.com/user-attachments/assets/af8fa48d-d8b1-4ead-89e6-3685aa8fcd15" />

<img width="927" height="315" alt="image" src="https://github.com/user-attachments/assets/506ee731-12c0-4014-9c69-0431f7075c2c" />


## Result:
Thus the program to implement the  Decision Tree Classifier Model for Predicting Employee Churn is written and verified using python programming.
