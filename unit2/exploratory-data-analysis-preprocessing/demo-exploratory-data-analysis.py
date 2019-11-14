import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv('HR_dataset.csv')


# First five rows
# print(df.head())

# DataFrame shape (rows, columns)
# print(df.shape)

# To find what all columns it contains, of what types and if they contain any value in it or not.
# print(df.info())

# To find the count, mean, standard deviation, minimum and maximum values and the quantities of the data.
# print(df.describe())

# Visualization
# sns.set()

# Check if there is any missing value
# sns.heatmap(df.isnull(), cmap='viridis')

# Check if there is any outlier in "Pay Rate" column
# sns.boxplot(df['Pay Rate'].values, orient='v')

# Display the figure
# plt.show()

# Check if there are any duplicates.
# Compare #rows in shape before and shape after.
# If there is any difference, then there are duplicates and they have been removed.
# print("Shape before:", df.shape, '\n------------')
# print(df.drop_duplicates())
# print("------------\nShape after:", df.shape)
