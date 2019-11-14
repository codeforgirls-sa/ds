import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import Imputer
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn.tree import export_graphviz  # Visualization

import warnings

# disable warnings
warnings.filterwarnings("ignore")

dataset = pd.read_csv('weatherAUS.csv')


# ----------------------------------------------- EDA & Preprocessing ----------------------------------------------- #

# check if there is any missing value
# sns.set()
# sns.heatmap(dataset.isnull(), cmap='viridis')
# plt.show()

# remove missing values in numeric columns
imputer = Imputer(missing_values='NaN', strategy='mean', axis=0)
df_num_col = ["MinTemp", "MaxTemp", "Rainfall", "Evaporation", "Sunshine", "WindGustSpeed", "WindSpeed9am",
              "WindSpeed3pm", "Humidity9am", "Humidity3pm", "Pressure9am", "Pressure3pm", "Cloud9am", "Cloud3pm",
              "Temp9am", "Temp3pm"]

data_num = dataset[df_num_col]
imputer = imputer.fit(data_num)
dataset[df_num_col] = imputer.transform(data_num)

# Use encoder for categorical variables
label_encoder = LabelEncoder()
df_cat_col = ["WindGustDir", "WindDir9am", "WindDir3pm", "RainToday", "RainTomorrow", "Date", "Location"]

# Update NaN values by NA
data_cat = dataset[df_cat_col].fillna('NA')

# Loop through each categorical variable and update values using LabelEncoder
# - remove missing values in categorical columns
for i in range(len(data_cat.columns)):
    data_cat.iloc[:, i] = label_encoder.fit_transform(data_cat.iloc[:, i])

dataset[df_cat_col] = data_cat

# check if there is any missing value
# sns.set()
# sns.heatmap(dataset.isnull(), cmap='viridis')
# plt.show()


# ----------------------------------------------- Feature Engineering ----------------------------------------------- #

# We need to remove RISK_MM because we want to predict 'RainTomorrow' and RISK_MM can leak some info to our model
dataset = dataset.drop('RISK_MM', axis=1)


# ----------------------------------------------- Predictive Modeling ----------------------------------------------- #

# split dataset in features and target variable
features = dataset.iloc[:, 0:22].values
target = dataset.iloc[:, 22].values

# Split dataset into training set and test set 70% training and 30% test
X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.3, random_state=0)

# Create Decision Tree classifier object
clf = DecisionTreeClassifier(random_state=0, max_depth=8)

# Train Decision Tree Classifier
clf = clf.fit(X_train, y_train)

# Predict the response for test dataset
y_pred = clf.predict(X_test)


# ---------------------------------------------- Predictive Evaluation ---------------------------------------------- #

print("Accuracy:", accuracy_score(y_test, y_pred)*100)


# ----------------------------------------------- Result Visualization ---------------------------------------------- #

export_graphviz(clf, out_file='tree_classifier.dot',
                feature_names=["Date", "Location", "MinTemp", "MaxTemp", "Rainfall", "Evaporation", "Sunshine",
                               "WindGustDir", "WindGustSpeed", "WindDir9am", "WindDir3pm", "WindSpeed9am",
                               "WindSpeed3pm", "Humidity9am", "Humidity3pm", "Pressure9am", "Pressure3pm", "Cloud9am",
                               "Cloud3pm", "Temp9am", "Temp3pm", "RainToday"],
                class_names=["Yes", "No"])

# Open Terminal:  dot -Tpng tree_classifier.dot -o tree_classifier.png => save it as png
# Samples = how many sample in the node
# Values = are class probabilities
