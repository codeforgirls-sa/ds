import pandas as pd

'''
Predicting the age of abalone from physical measurements. The age of abalone is determined by cutting the shell
through the cone, staining it, and counting the number of rings through a microscope -- a boring and time-consuming
task. Other measurements, which are easier to obtain, are used to predict the age. Further information,
such as weather patterns and location (hence food availability) may be required to solve the problem.
'''

snails = pd.read_csv('abalone.csv')


# Create new feature age = Rings + 1.5
snails['age'] = snails['Rings'] + 1.5
# print(snails.head())


# Create categories from age column
def create_age_categories(age):
    if age < 9.5:
        return 'Young'
    elif age > 12.5:
        return 'Old'
    else:
        return 'Middle-aged'


# Pandas.apply allow the users to pass a function and apply it on every single value of the Pandas series.
snails['categorical_age'] = snails['age'].apply(create_age_categories)
# print(snails.head())


# What if our algorithm needs categories expressed as integer? => Convert string categories into integers!
age_to_int = {'Young': 1, 'Middle-aged': 2, 'Old': 3}

snails['categorical_age_int'] = snails['categorical_age'].map(age_to_int)
# print(snails.head())
