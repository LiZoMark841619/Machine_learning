import pandas as pd 
import numpy as np 
import seaborn as sns
import matplotlib.pyplot as plt  
from sklearn.linear_model import LinearRegression

df = pd.read_csv("honey.csv")
print(df.head())
prod_per_year = df.groupby('year').totalprod.mean().reset_index()
print(prod_per_year)
X = prod_per_year.year.values.reshape(-1,1)
print(type(X))
y = prod_per_year.totalprod
print(type(y))
regr = LinearRegression()
regr.fit(X, y)
print(regr.coef_[0], regr.intercept_)
y_predict = regr.predict(X)

plt.scatter(x=X, y=y)
plt.plot(X, y_predict, '-', color='red')
plt.show()
plt.clf()
X_future = np.array(range(2013, 2051))
X_future = X_future.reshape(-1, 1)
future_predict = regr.predict(X_future)
plt.plot(X_future, future_predict, '-', color='green')
plt.show()
plt.clf()

