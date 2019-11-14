import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error
from sklearn.tree import export_graphviz  # Visualization


dataset = pd.read_csv('Boston.csv')

features = dataset.iloc[:, 6:7].values
target = dataset.iloc[:, 14].values

X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.3, random_state=0)

max_depth_range = list(range(1, 10))
# List to store the average RMSE for each value of max_depth:
accuracy = []
for depth in max_depth_range:
    clf = DecisionTreeRegressor(max_depth=depth,
                                 random_state=0)
    clf.fit(X_train, y_train)
    score = clf.score(X_test, y_test)
    accuracy.append(score)

sns.set()
sns.lineplot(max_depth_range, accuracy)
plt.show()
