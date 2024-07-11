import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error
import math

data = pd.read_csv('../data/data')
print(data.head())
print(data.describe())
print(data.info())
print(data.columns)

#Trying to find Correlation between columns
("""
sns.jointplot(x='Time on Website', y='Yearly Amount Spent', data=data, alpha=0.5)

sns.pairplot(data, kind='scatter', plot_kws={'alpha': 0.4})

sns.jointplot(x='Time on App', y='Yearly Amount Spent', data=data, alpha=0.5)

sns.lmplot(x='Length of Membership',
           y='Yearly Amount Spent',
           data=data,
           scatter_kws={'alpha': 0.4}
           )
plt.show()
""")

#train-test

x = data[['Avg. Session Length', 'Time on App', 'Time on Website', 'Length of Membership']]
y = data['Yearly Amount Spent']

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=42)

#training model
lm = LinearRegression()
lm.fit(x_train, y_train)

cdf = pd.DataFrame(lm.coef_, x.columns, columns=['Coef'])
print(cdf)

#predictions

predictions = lm.predict(x_test)


sns.scatterplot(x= predictions, y =y_test )
plt.xlabel("Predictions")
plt.title("Evalution of our LM model")
plt.show()

print("Mean Absolute Error: ", mean_absolute_error(y_test,predictions))
print("Mean Squared Error: ", mean_squared_error(y_test,predictions))
print("RMSE: ",math.sqrt(mean_squared_error(y_test,predictions)))

#residuals

residuals = y_test - predictions
sns.displot(residuals,bins = 10,kde = True)
plt.show()

