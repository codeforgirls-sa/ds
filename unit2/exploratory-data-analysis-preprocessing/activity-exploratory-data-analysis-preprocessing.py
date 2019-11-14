import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


list1 = [200, 300, 3400, 500, 100, 600, -1050, 700, 800]

# Convert into DataFrame
df = pd.DataFrame(list1)
df = df.clip(100, 900)

# Outliers
sns.set()
sns.boxplot(df, orient='v')
plt.show()
