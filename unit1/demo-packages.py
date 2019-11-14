import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# ----------------------------------------------------- Numpy ------------------------------------------------------- #

# In NumPy dimensions are called axes.
# The list arr has 2 axes. The first axis has a length of 2, the second axis has a length of 3.
list1 = [[1., 0., 0.], [0., 1., 2.]]

# Convert list1 to numpy array.
arr = np.asarray(list1)

# Create numpy array
arr2 = np.array([1, 2, 3, 4])

# Print the number of axes (dimensions) of the array.
# print("Number of axes:", arr.ndim)

# Print #rows and #columns. For a matrix with n rows and m columns, shape will be (n, m).
# print("Shape:", arr.shape)

# Print the total number of elements of the array. This is equal to the product of the elements of shape.
# print("Total number of elements:", arr.size)

# Print an object describing the type of the elements in the array.
# print("The type of the elements in the array:", arr.dtype)

# Square root
s = 25
# print("Square root of 25 is:", np.sqrt(s))

# Power
p = 6
# print("6 to the power 2 is:", np.power(p, 2))

# ----------------------------------------------------- Pandas ------------------------------------------------------ #

# Convert from list to pandas DataFrame
numbers = [3, 4, 5, 6, 7]
newDF = pd.DataFrame(numbers)
# print("New DataFrame:")
# print(newDF)

# Create pandas DataFrame.
df = pd.DataFrame([[1, 2, 3], [4, 5, 6], [7, 8, 9], [10, 11, 12], [13, 14, 15], [16, 17, 18]])

# To view a small sample of a DataFrame object, use the head() and tail().
# Default number of elements to display is 5, but you may pass a custom number.
# print("Head of df:")
# print(df.head())

# print("\nHead of df with custom number:")
# print(df.head(2))

# print("\nTail of df:")
# print(df.tail())

# print("\nTail of df with custom number:")
# print(df.tail(2))

# Print shape of df
# print("Shape:", df.shape)

# Print from index 0 to 2 (2 is excluded) => 0 to 1
# print(df[0:2], end='\n\n')

# OR (if we didn't specify the first index it means 0)
# print(df[:2])

# Print all rows except last 2
# print(df[:-2])

# To find what all columns it contains, of what types and if they contain any value in it or not. Use info()
# print(df.info())

# To find the count, mean, standard deviation, minimum and maximum values and the quantities of the data. Use describe()
# print(df.describe())

# To remove a column
# print(df.drop(columns=[1], axis=1))

# To remove a row
# print(df.drop(index=0, axis=0))

# Create new DataFrame with duplicate rows
df2 = pd.DataFrame([[1, 2, 3], [1, 2, 3], [4, 5, 6], [4, 5, 6], [7, 8, 9], [7, 8, 9]])

# To remove duplicate rows use drop_duplicates()
df2 = df2.drop_duplicates()
# print(df2)

# Read csv file (dataset)
df3 = pd.read_csv('hotaling_cocktails - Cocktails.csv')

# ----------------------------------------------------- Sklearn ----------------------------------------------------- #

# There are many uses of sklearn:
# fit() to train the model
# predict() to test the model
# to evaluate the model
# to visualize the results
# to split the dataset into training and testing
# to use different classifiers

# ----------------------------------------------------- Seaborn ----------------------------------------------------- #

# We can use seaborn also to create statistical data visualization
# Seaborn built on top of the Matplotlib library. Seaborn is fast and easy to learn.

# Set background color
# sns.set(style='darkgrid')

# Create boxplot for df ([[1, 2, 3], [4, 5, 6], [7, 8, 9], [10, 11, 12], [13, 14, 15], [16, 17, 18]])
# orient = 'v' => set the orientation as vertical, the default is horizontal
# sns.boxplot(df, orient='v')

# Display the figure
# plt.show()
