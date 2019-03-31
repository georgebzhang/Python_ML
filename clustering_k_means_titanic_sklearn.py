import matplotlib.pyplot as plt
from matplotlib import style
import numpy as np
from sklearn.cluster import KMeans
from sklearn import preprocessing, model_selection
import pandas as pd
import xlrd
style.use('ggplot')

'''
pclass: passenger class (1 = 1st, 2 = 2nd, 3 = 3rd)
survived: (0 = no, 1 = yes)
name:
sex: female, male
age:
sibsp: # of siblings and spouses aboard
parch: # of parents and children aboard
ticket: ticket #
fare: in British pounds
cabin:
embarked: port of embarkation (C = Cherbourg, Q = Queenstown, S = Southampton)
boat: lifeboat
body: body identification #
home.dest: home/destination
'''


def handle_non_numerical_data(df):
    columns = df.columns.values

    for column in columns:
        text_digit_vals = {}

        def convert_to_int(val):
            return text_digit_vals[val]

        if df[column].dtype != np.int64 and df[column].dtype != np.float64:
            column_contents = df[column].values.tolist()
            unique_elements = set(column_contents)
            x = 0

            for element in unique_elements:
                if element not in text_digit_vals:
                    text_digit_vals[element] = x
                    x += 1

            df[column] = list(map(convert_to_int, df[column]))
            # df[column] = [convert_to_int(val) for val in df[column]]

    return df

df = pd.read_excel('titanic.xls')
df.drop(['body', 'name'], 1, inplace=True)
df.convert_objects(convert_numeric=True)
df.fillna(0, inplace=True)
df = handle_non_numerical_data(df)
print(df.head())