# code run
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler

df=pd.read_csv("C:/Engineering/Thirdyear/MLpractical/Mall_Customers.csv")
df1=df.drop('Genre',axis=1)

corr=df1.corr()
sns.heatmap(corr,fmt='.1f',annot=True)
plt.show()



plt.scatter(df['Age'],df['Annual Income (k$)'])
plt.show()

from sklearn.cluster import KMeans



x=df.iloc[:,[2,3,4]].values


sse=[]
for i in range(1,30):
    km=KMeans(n_clusters=i)
    km.fit(x)
    sse.append(km.inertia_)

print(sse)
plt.plot(range(1,30),sse)
plt.show()

km=KMeans(n_clusters=5)
y=km.fit_predict(x)
print(y)
df['cluster']=y
print(df.head())
from sklearn.preprocessing import MinMaxScaler
scaler=MinMaxScaler()
scaler.fit(df[['Age']])
df['Age']=scaler.transform(df[['Age']])

scaler.fit(df[['Annual Income (k$)']])
df['Annual Income (k$)']=scaler.transform(df[['Annual Income (k$)']])
df2=df[df.cluster==0]
df3=df[df.cluster==1]
df4=df[df.cluster==2]
df5=df[df.cluster==3]
df6=df[df.cluster==4]

plt.scatter(df2.Age,df2['Annual Income (k$)'],color='red')
plt.scatter(df3.Age,df3['Annual Income (k$)'],color='blue')
plt.scatter(df4.Age,df4['Annual Income (k$)'],color='green')
plt.scatter(df5.Age,df5['Annual Income (k$)'],color='yellow')
plt.scatter(df6.Age,df6['Annual Income (k$)'],color='black')
plt.show()