# flake8: noqa
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    classification_report,
)
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier as rfc
from sklearn.preprocessing import StandardScaler

# Read data from the csv file
# Source: https://www.kaggle.com/datasets/radheshyamkollipara/bank-customer-churn/data

churn_data = pd.read_csv("Customer-Churn-Records.csv")
churn_data.head()

# 1.check for missing values
churn_data.isnull().sum().sum()

# 2.check for duplicate values
churn_data.duplicated().sum()

# drop the unnecessary columns
churn_data.drop(columns=["RowNumber", "CustomerId", "Surname"], inplace=True)
churn_data.head()

# info about data types and non-null values
churn_data.info()

churn_data.describe()

# Explore the distribution of gender
plt.figure(figsize=(5, 3))
sns.countplot(
    x="Gender",
    data=churn_data,
    palette={"Female": "yellow", "Male": "lightblue"},
)
plt.title("Distribution of Gender")
plt.xlabel("Female = 0 , Male = 1")
plt.ylabel('Number of Customers')
plt.show()

# Explore the distribution of age
plt.figure(figsize=(5, 3))
sns.histplot(x="Age", data=churn_data, bins=25)
plt.ylabel("Number of Customers")
plt.title("Distribution of Age")
plt.show()

# Explore if the customer has credit card?
plt.figure(figsize=(5, 3))
sns.countplot(x="HasCrCard", data=churn_data, palette={"0": "red", "1": "green"})
plt.title(" Customer Has a Credit Card ?")
plt.xlabel(" No = 0 , Yes = 1 ")
plt.ylabel("Number of Customers")
plt.show()

# Check if there's any relation between individual columns and the output.
# eg.See if gender has any effect on customer churn.

# 1.Gender and Customer Churn Relationship
# target variable : Exited

counts = churn_data.groupby(["Gender", "Exited"]).Exited.count().unstack()
gender_exited = counts.plot(kind="bar", stacked=True)
gender_exited.set_xlabel("Gender")
gender_exited.set_ylabel("Number of Customers")
gender_exited

# From the visualization, Female customers left the bank more often compared to the Male customers.
churn_data.head()


# 2. Geography and Customer Churn Relationship
# target variable : Exited

counts = churn_data.groupby(["Geography", "Exited"]).Exited.count().unstack()
geo_exited = counts.plot(kind="bar", stacked=True)
geo_exited.set_xlabel("Geography")
geo_exited.set_ylabel("Number of Customers")
geo_exited


# Since Geography and Gender columns are categorical columns ,
# We convert categorical data into numerical data using one-hot encoding scheme.

# FOR regression model remove the categorical column and add one column for each of the unique values in the removed column.
# Then add 1 to the column where the actual value existed and add 0 to the rest of the columns.

# 1. Remove the Geography column

# 2. Add a column for each of three unique values- France, Germany, and Spain.

# 3. Then add 1 in the column where actually value existed.
# Logic: If the first record contained Spain in the original Geography column, we will add 1 in the Spain column
#        and zeros in the columns for France and Germany.)

# --------------------------Data Preprocessing--------------------------------------------------------------------

# 1. Drop columns "Geography" and "Gender"

temp = churn_data.drop(["Geography", "Gender", "Card Type"], axis=1)
temp

# 2 : Create one hot coded columns for 'Geography' and 'Gender' and 'Card Type'

# get_dummies() converts categorical value into binary(0 or 1),
# iloc[:,1:] - all rows, but drops first column to avoid redundancy


Geography = pd.get_dummies(churn_data.Geography).iloc[:, 1:]
Gender = pd.get_dummies(churn_data.Gender).iloc[:, 1:]
CardType = pd.get_dummies(churn_data["Card Type"]).iloc[:, 1:]

# 3 : Concatenate our churn data set with the one hot encoded vectors

churn_data = pd.concat([temp, Geography, Gender, CardType], axis=1)
churn_data.head()


# ------------------------------------Model Training and Test sets------------------------------------------------

# Training set: To train the model (ML algorithm)
# Test Set: Model will be evaluated on test set.

# 1 : Divide the data into labels and feature set.

# feature set - all the columns except 'Exited'. Since value in the column 'Exited' is to be predicted.
# label set   - 'Exited' column

features = churn_data.drop(["Exited"], axis=1)
labels = churn_data["Exited"]

# Check if we still have any categorical values
features.select_dtypes(include="object").columns

# Divide training and test

# test will consist of 20% of the total dataset.
# using train_test_split from package - sklearn.model_selection   (import statement at the beginning)
# train_test_split shuffles the data and outputs 4 arrays/dataframes
# input args - features: churn_data excluding exited, label: target column, test_size: 20% for test,
# random_state= seed to ensure reproducibilty

train_features, test_features, train_labels, test_labels = train_test_split(
    features, labels, test_size=0.2, random_state=20
)

# 1. Random forest
# n_estimators:Number of decision trees in the forest.
#             More trees → usually better performance but slower training
# fit() - trains the model, learns patters in training data.
# predict() - predicts labels for unseen data, returns an array 'predicted_lables' with values 0/1 for churn,
# compared to actual test labels to evaluate performance.

rfc_object = rfc(n_estimators=200, random_state=10)

rfc_object.fit(train_features, train_labels)

predicted_labels = rfc_object.predict(test_features)

# The most commonly used metrics are precision and recall, F1 measure, accuracy and confusion matrix.
# The Scikit Learn library contains classes that can be used to calculate these metrics.

# from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
# Evaluate performance

print("Classification Report: \n",classification_report(test_labels, predicted_labels))

print("Confusion Matrix: \n",confusion_matrix(test_labels, predicted_labels))

print("Accuracy score: \n",accuracy_score(test_labels, predicted_labels))

# Random Forest model accuracy of 99.8 for customer churn prediction.

# Note: Training on another model since above results indicate overfitting of data.

# 2.Logistic regression:

# from sklearn.linear_model import LogisticRegression
scalar = StandardScaler()

train_scaled = pd.DataFrame(scalar.fit_transform(train_features), columns= train_features.columns)
test_scaled = pd.DataFrame(scalar.fit_transform(test_features), columns= test_features.columns)

lr = LogisticRegression(solver='saga',max_iter=1000,random_state=20)

lr.fit(train_scaled, train_labels)

predicted_labels = lr.predict(test_scaled)

# Evaluate performance:

print("Classification Report: \n", classification_report(test_labels, predicted_labels))

print("Confusion Matrix: \n", confusion_matrix(test_labels, predicted_labels))

print("Accuracy score: \n", accuracy_score(test_labels, predicted_labels))

# Logistic Regression model accuracy: 99.85%
