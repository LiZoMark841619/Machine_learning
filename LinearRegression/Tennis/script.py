import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import seaborn as sns

data = pd.read_csv('tennis_stats.csv')
df = pd.DataFrame(data)

print(df.head(3))
print(df.shape)
print(df.info())

plt.figure(figsize=[14, 14])
sns.heatmap(df[df.columns.to_list()[7:]].corr(), annot=True, cmap='coolwarm')
plt.show()
plt.clf()


columns = df.columns.to_list()[7:23]

for feature in columns:
    plt.style.use('fivethirtyeight')
    plt.figure(figsize=[8, 5])
    sns.scatterplot(x='Wins', y=feature, data=df, hue='Year', palette='rocket')
    plt.title(f'Scatter plot of {df.columns.to_list()[-4]} and {feature}')
    plt.show()
    plt.clf()
    plt.figure(figsize=[8, 5])
    sns.scatterplot(x='Winnings', y=feature, data=df, hue='Year', palette='rocket')
    plt.title(f'Scatter plot of {df.columns.to_list()[-2]} and {feature}')
    plt.show()
    plt.clf()
    
X1 = df[['Aces', 'ReturnGamesPlayed']]
y = df[['Wins']]
x_train1, x_test1, y_train1, y_test1 = train_test_split(X1, y, train_size=0.8, test_size=0.2, random_state=6)
mlr1 = LinearRegression()

model1 = mlr1.fit(x_train1, y_train1)
y_predict1 = model1.predict(x_test1)
print(mlr1.coef_)

print(mlr1.score(x_train1, y_train1))
print(mlr1.score(x_test1, y_test1))

plt.figure(figsize=[8, 5])
plt.scatter(y_test1, y_predict1)
plt.title('Actual and predicted Wins - Feature 2')
plt.xlabel('Actual number of Wins')
plt.ylabel('Predicted number of Wins')
plt.show()
plt.clf()


X2 = df[['Aces', 'BreakPointsFaced', 'BreakPointsOpportunities', 'DoubleFaults', 'ReturnGamesPlayed', 'ServiceGamesPlayed', 'Losses']]
y = df[['Wins']]
x_train2, x_test2, y_train2, y_test2 = train_test_split(X2, y, train_size = 0.8, test_size = 0.2, random_state=6)
mlr2 = LinearRegression()
model2 = mlr2.fit(x_train2, y_train2)
y_predict2 = mlr2.predict(x_test2)

print(mlr2.score(x_train2, y_train2))
print(mlr2.score(x_test2, y_test2))

plt.figure(figsize=[8, 5])
plt.scatter(y_test2, y_predict2)
plt.title('Actual and predicted Wins - Feature 7')
plt.xlabel('Actual number of Wins')
plt.ylabel('Predicted number of Wins')
plt.show()
plt.clf()