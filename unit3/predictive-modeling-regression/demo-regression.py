import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error
from sklearn.tree import export_graphviz  # Visualization


dataset = pd.read_csv('Boston.csv')


# ----------------------------------------------- EDA & Preprocessing ----------------------------------------------- #

# check if there is any missing value
# sns.set()
# sns.heatmap(dataset.isnull(), cmap='viridis')
# plt.show()


# ----------------------------------------------- Feature Engineering ----------------------------------------------- #

# split dataset in features and target variable
features = dataset.iloc[:, 6:7].values
target = dataset.iloc[:, 14].values


# ----------------------------------------------- Predictive Modeling ----------------------------------------------- #

# Split dataset into training set and test set 70% training and 30% test
X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.3, random_state=0)

# Create Decision Tree Regressor object
clf = DecisionTreeRegressor(max_depth=3)

# Train Decision Tree Regressor
clf.fit(X_train, y_train)

# Predict the response for test dataset
y_pred = clf.predict(X_test)


# ---------------------------------------------- Predictive Evaluation ---------------------------------------------- #

print("Mean Squared Error (MSE): ", mean_squared_error(y_test, y_pred))
print("Root Mean Squared Error (RMSE): ", np.sqrt(mean_squared_error(y_test, y_pred)))


# ----------------------------------------------- Result Visualization ---------------------------------------------- #

export_graphviz(clf, out_file='tree_regressor.dot',
                feature_names=["rm"], filled=True)

# Open Terminal:  dot -Tpng tree_regressor.dot -o tree_regressor.png => save it as png
# Samples = how many sample in the node
# Value =
