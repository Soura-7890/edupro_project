import pandas as pd
import numpy as np

file = "edupro.xlsx"

# Load sheets
courses = pd.read_excel(file, sheet_name="Courses")
teachers = pd.read_excel(file, sheet_name="Teachers")
transactions = pd.read_excel(file, sheet_name="Transactions")

# -------------------------------
# CORRECT MERGING
# -------------------------------

# Merge Transactions + Courses
df = pd.merge(transactions, courses, on="CourseID", how="left")

# Merge with Teachers
df = pd.merge(df, teachers, on="TeacherID", how="left")

print("\nMerged Data:\n", df.head())

# -------------------------------
# CLEANING
# -------------------------------

df = df.dropna()

df["TransactionDate"] = pd.to_datetime(df["TransactionDate"])

# -------------------------------
# FEATURE ENGINEERING
# -------------------------------

# Enrollment count per course
enrollment = df.groupby("CourseID").size().reset_index(name="EnrollmentCount")

df = pd.merge(df, enrollment, on="CourseID", how="left")

# Price band
df["PriceBand"] = pd.cut(df["CoursePrice"], bins=3, labels=["Low", "Medium", "High"])

# Duration bucket
df["DurationBucket"] = pd.cut(df["CourseDuration"], bins=3)

# -------------------------------
# MODEL
# -------------------------------

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score

features = [
    "CoursePrice",
    "CourseDuration",
    "CourseRating",
    "TeacherRating",
    "YearsOfExperience"
]

target = "EnrollmentCount"

df = df.dropna(subset=features + [target])

X = df[features]
y = df[target]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Linear Regression
lr = LinearRegression()
lr.fit(X_train, y_train)

pred_lr = lr.predict(X_test)

print("\nLinear Regression")
print("MAE:", mean_absolute_error(y_test, pred_lr))
print("R2:", r2_score(y_test, pred_lr))

# Random Forest
rf = RandomForestRegressor()
rf.fit(X_train, y_train)

pred_rf = rf.predict(X_test)

print("\nRandom Forest")
print("MAE:", mean_absolute_error(y_test, pred_rf))
print("R2:", r2_score(y_test, pred_rf))

# -------------------------------
# FEATURE IMPORTANCE
# -------------------------------

import matplotlib.pyplot as plt

importance = rf.feature_importances_

plt.bar(features, importance)
plt.title("Feature Importance")
plt.show()
import seaborn as sns
import matplotlib.pyplot as plt

# -------------------------------
# EDA (Exploratory Data Analysis)
# -------------------------------

# Enrollment distribution
plt.figure()
sns.histplot(df["EnrollmentCount"], bins=30)
plt.title("Enrollment Distribution")
plt.show()

# Price vs Enrollment
plt.figure()
sns.scatterplot(x=df["CoursePrice"], y=df["EnrollmentCount"])
plt.title("Price vs Enrollment")
plt.show()

# Rating vs Enrollment
plt.figure()
sns.scatterplot(x=df["CourseRating"], y=df["EnrollmentCount"])
plt.title("Rating vs Enrollment")
plt.show()

# Experience vs Enrollment
plt.figure()
sns.scatterplot(x=df["YearsOfExperience"], y=df["EnrollmentCount"])
plt.title("Instructor Experience vs Enrollment")
plt.show()
# -------------------------------
# Revenue Prediction
# -------------------------------

df["Revenue"] = df["Amount"]

revenue = df.groupby("CourseID")["Revenue"].sum().reset_index()

df = pd.merge(df, revenue, on="CourseID", how="left")

target = "Revenue"

X = df[features]
y = df[target]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

rf.fit(X_train, y_train)
pred = rf.predict(X_test)

print("\nRevenue Prediction")
print("MAE:", mean_absolute_error(y_test, pred))
print("R2:", r2_score(y_test, pred))
