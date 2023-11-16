#code run
import pandas as pandas
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

df=pandas.read_csv("C:/Engineering/Thirdyear/MLpractical/temperatures.csv")
print(df.size)
print(df.shape)
print(df.describe())

df.plot(x='JAN',y='FEB',style='o');
plt.title('jan vs Feb')
plt.xlabel('Min. Temp.')
plt.ylabel('Max. Temp.')
plt.show()

plt.figure(figsize=(15,10))
sns.displot(df['FEB'])
plt.show()

from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
x=df['JAN'].values.reshape(-1,1)
y=df['FEB'].values.reshape(-1,1)
xtrain,xtest,ytrain,ytest=train_test_split(x,y,test_size=0.2,random_state=0)
model=LinearRegression()
model.fit(xtrain,ytrain)
print("Intercept is = ",model.intercept_)
print("coefficient is = ",model.coef_)
y_pred=model.predict(xtest)


df2=pandas.DataFrame({'Actual':ytest.flatten(),'Predicted':y_pred.flatten()})
print(df2)
df3=df2.head(25)
df3.plot(kind='bar',figsize=(20,20))
plt.show()

plt.scatter(xtest,ytest)
plt.plot(xtest,y_pred,linewidth=3)
plt.show()

from sklearn import metrics
print("mean absolute error = ",metrics.mean_absolute_error(ytest,y_pred))
print("mean squared error = ",metrics.mean_squared_error(ytest,y_pred))
print("root mean squared error = ",np.sqrt(metrics.mean_squared_error(ytest,y_pred)))