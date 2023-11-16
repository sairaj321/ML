# code run
import pandas as pd
import numpy as np
import matplotlib.pyplot as mp
import seaborn as sns
import statistics as s

df=pd.read_csv("C:/Engineering/Thirdyear/MLpractical/heart1.csv")
print(df.head())

print(df.info())

print("size of dataset ",df.size)

print("shape of dataset = ",df.shape)

print(df.describe())

print("data types for each column\n",df.dtypes.value_counts())

n=df.columns[df.dtypes=='object']
print(df[n].isnull().sum())

print(df[n].isnull().sum().sort_values(ascending=False)/len(df))

print("mean=",s.mean(df['Age']))

print(df["Sex"])

from sklearn.model_selection import train_test_split
col_delete=['HeartDisease','Sex','ChestPainType','RestingECG','ExerciseAngina','ST_Slope']
x=df.drop(col_delete,axis=1)
y=df.HeartDisease
xtrain,xtest,ytrain,ytest=train_test_split(x,y,test_size=0.2,random_state=42)
print("train dataset\n",xtest)
print("\n\nTest dataset\n",ytest)

from sklearn.linear_model import LogisticRegression

model = LogisticRegression()

# Fit the model to the training data
model.fit(xtrain, ytrain)

# Make predictions on the test set
y_pred = model.predict(xtest)




from sklearn.metrics import confusion_matrix,ConfusionMatrixDisplay

print(model.classes_)

cm=confusion_matrix(ytest,y_pred,labels=model.classes_)
pl=ConfusionMatrixDisplay(confusion_matrix=cm,display_labels=['Diseased','Not Diseased'])
pl.plot()
mp.show()

from sklearn.neighbors import KNeighborsClassifier

knn=KNeighborsClassifier(n_neighbors=3)
knn.fit(xtrain,ytrain)
ks=knn.score(xtest,ytest)
print(ks)

cm1=confusion_matrix(ytest,y_pred,labels=model.classes_)
cd=ConfusionMatrixDisplay(cm1,display_labels=model.classes_)
cd.plot()
mp.show()

from sklearn.metrics import classification_report
print(classification_report(ytest,y_pred))