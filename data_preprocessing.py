import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# ------------------------------
# 1. Load Dataset
# ------------------------------
# Titanic dataset (can be replaced with any dataset)
url = "https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv"
df = pd.read_csv(url)

print("===== Basic Info =====")
print(df.info())
print("\n===== Missing Values =====")
print(df.isnull().sum())

# ------------------------------
# 2. Handle Missing Values
# ------------------------------
# Fill Age with median
df['Age'].fillna(df['Age'].median(), inplace=True)

# Fill Embarked with mode
df['Embarked'].fillna(df['Embarked'].mode()[0], inplace=True)

# Drop Cabin due to too many missing values
df.drop(columns=['Cabin'], inplace=True)

# ------------------------------
# 3. Encode Categorical Variables
# ------------------------------
# Convert Sex into numerical (0/1)
df['Sex'] = df['Sex'].map({'male': 0, 'female': 1})

# One-hot encode Embarked
df = pd.get_dummies(df, columns=['Embarked'], drop_first=True)

# ------------------------------
# 4. Normalize / Standardize Numerical Features
# ------------------------------
from sklearn.preprocessing import StandardScaler

num_features = ['Age', 'Fare']
scaler = StandardScaler()
df[num_features] = scaler.fit_transform(df[num_features])

# ------------------------------
# 5. Outlier Detection & Removal
# ------------------------------
plt.figure(figsize=(10,5))
sns.boxplot(data=df[num_features])
plt.title("Boxplot Before Outlier Removal")
plt.show()

# Remove outliers using IQR method
for col in num_features:
    Q1 = df[col].quantile(0.25)
    Q3 = df[col].quantile(0.75)
    IQR = Q3 - Q1
    lower = Q1 - 1.5 * IQR
    upper = Q3 + 1.5 * IQR
    df = df[(df[col] >= lower) & (df[col] <= upper)]

plt.figure(figsize=(10,5))
sns.boxplot(data=df[num_features])
plt.title("Boxplot After Outlier Removal")
plt.show()

# ------------------------------
# Final Cleaned Dataset
# ------------------------------
print("\n===== Cleaned Data Sample =====")
print(df.head())

# Save cleaned dataset for ML tasks
df.to_csv("cleaned_titanic.csv", index=False)
print("\nCleaned dataset saved as cleaned_titanic.csv")
