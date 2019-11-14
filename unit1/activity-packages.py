import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# Numpy - Find square root of 1024 => 32
print(np.sqrt(1024))


# Pandas - Convert list1 to DataFrame and remove the duplicates
list1 = [1, 2, 3, 4, 5, 6, 7, 8, 8, 9, 10]

df = pd.DataFrame(list1)
print("Shape before: ", df.shape)  # (11, 1)
df = df.drop_duplicates()
print("Shape after: ", df.shape)  # (10, 1)


# Seaborn - Visualize df using boxplot
sns.set(style='darkgrid')
sns.boxplot(df)
plt.show()
