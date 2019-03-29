import pandas as pd
import quandl
import math
import numpy as np
from sklearn import preprocessing, model_selection, svm
from sklearn.linear_model import LinearRegression
import datetime
import matplotlib.pyplot as plt
from matplotlib import style
import pickle

style.use('ggplot')

# df: dataframe
df = quandl.get('WIKI/GOOGL')  # get dataframe from quandl
df = df[['Adj. Open', 'Adj. High', 'Adj. Low', 'Adj. Close', 'Adj. Volume']]  # redefine dataframe keeping only Adj. Open, Adj. High, Adj. Low, Adj. Close, Adj. Volume columns
# define new columns
df['HL_PCT'] = (df['Adj. High'] - df['Adj. Close']) / df['Adj. Close'] * 100.0  # percent volatility (high % - low %)
df['PCT_change'] = (df['Adj. Close'] - df['Adj. Open']) / df['Adj. Open'] * 100.0  # daily percent change
df = df[['Adj. Close', 'HL_PCT', 'PCT_change', 'Adj. Volume']]  # redefine dataframe keeping only Adj. Close, HL_PCT, PCT_change, Adj. Volume columns (features)

forecast_col = 'Adj. Close'
df.fillna(-99999, inplace=True)  # fill NA/NaN values with outlier -99999

# regression generally used to forecast out
forecast_out = int(math.ceil(0.01*len(df)))  # predict out 1% of dataframe
print('Forcasting {} days in advance:'.format(forecast_out))

df['label'] = df[forecast_col].shift(-forecast_out)

X = np.array(df.drop(['label'], 1))  # features, axis=1 indicates dropped column
X = preprocessing.scale(X)  # for training on fixed set (e.g. not for high-frequency trading)
X = X[:-forecast_out]
X_lately = X[-forecast_out:]

df.dropna(inplace=True)  # remove missing values
y = np.array(df['label'])  # labels

# create training and testing sets
X_train, X_test, y_train, y_test = model_selection.train_test_split(X, y, test_size=0.2)  # 0.2 indicates 20% of data

clf = LinearRegression(n_jobs=-1)  # can specify n_jobs= for how many jobs/threads to run, n_jobs=-1 will run as many jobs as possible by processor
# clf = svm.SVR()  # support vector regression, can specify kernel=
clf.fit(X_train, y_train)
with open('linearregression.pickle', 'wb') as f:
    pickle.dump(clf, f)

# can comment out previous 4 lines (skip training) with pickle file
pickle_in = open('linearregression.pickle', 'rb')
clf = pickle.load(pickle_in)

accuracy = clf.score(X_test, y_test)  # accuracy is squared error for linear regression
print('Accuracy: {}'.format(accuracy))

forecast_set = clf.predict(X_lately)
print(forecast_set, accuracy, forecast_out)

df['Forecast'] = np.nan
last_date = df.iloc[-1].name
last_unix = last_date.timestamp()
one_day = 86400  # seconds
next_unix = last_unix + one_day  # next day

# to have dates on x-axis and add Forecast column with NaN near head and predictions near tail
for i in forecast_set:
    next_date = datetime.datetime.fromtimestamp(next_unix)
    next_unix += one_day
    df.loc[next_date] = [np.nan for _ in range(len(df.columns)-1)] + [i]

df['Adj. Close'].plot()
df['Forecast'].plot()
plt.legend(loc=4)
plt.xlabel('Date')
plt.ylabel('Price')
plt.show()