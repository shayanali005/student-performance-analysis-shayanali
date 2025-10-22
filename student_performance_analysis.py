# 🎓 Student Performance Data Analysis Project
# Author: Shayan Ali

# Step 1: Import Libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Step 2: Load Dataset
data = pd.read_csv("StudentsPerformance.csv")

# Step 3: Basic Information
print("\n--- Dataset Info ---")
print(data.info())

print("\n--- First 5 Rows ---")
print(data.head())

print("\n--- Missing Values ---")
print(data.isnull().sum())

# Step 4: Clean Column Names
data.columns = [col.strip().replace(" ", "_").lower() for col in data.columns]

# Step 5: Create Average Score Column
data['average_score'] = data[['math_score', 'reading_score', 'writing_score']].mean(axis=1)

# Step 6: Describe Dataset
print("\n--- Descriptive Statistics ---")
print(data.describe())

# Step 7: Visualizations
sns.set(style="whitegrid")

# Gender Distribution
sns.countplot(x='gender', data=data)
plt.title("Number of Male and Female Students")
plt.show()

# Test Preparation vs Math Score
sns.boxplot(x='test_preparation_course', y='math_score', data=data)
plt.title("Math Scores vs Test Preparation Course")
plt.show()

# Parental Education vs Average Score
sns.barplot(x='parental_level_of_education', y='average_score', data=data)
plt.xticks(rotation=45)
plt.title("Average Score by Parental Education")
plt.show()

# Step 8: Correlation Heatmap
sns.heatmap(data[['math_score', 'reading_score', 'writing_score', 'average_score']].corr(),
            annot=True, cmap='coolwarm')
plt.title("Correlation Between Scores")
plt.show()

# Step 9: Simple Predictive Model
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score

# Target column: Above Average Math Score
data['above_avg_math'] = (data['math_score'] >= 70).astype(int)

# Encode categorical columns
le = LabelEncoder()
for col in data.select_dtypes(include='object').columns:
    data[col] = le.fit_transform(data[col])

# Split data
X = data.drop(['math_score', 'above_avg_math', 'average_score'], axis=1)
y = data['above_avg_math']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)
preds = model.predict(X_test)

# Step 10: Evaluate Model
print("\n--- Model Accuracy ---")
print(f"Accuracy: {accuracy_score(y_test, preds):.2f}")
