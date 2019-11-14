import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv('HR_dataset.csv')


# Remove missing values.
# Since the missing value here found in "Date of Termination" column, we can't apply any statistical calculation.
# The solutions are either to drop this column or fill it manually.
df = df.drop(columns='Date of Termination', axis=1)

# sns.set()

# Check if there is any missing value
# sns.heatmap(df.isnull(), cmap='viridis')

# Remove outliers
df['Pay Rate'] = df['Pay Rate'].clip(14, 77)

# sns.boxplot(df['Pay Rate'].values, orient='v')

# plt.show()

# If there were any duplicates, we can do the following: df = df.drop_duplicates()
# => remove them then update the DataFrame
