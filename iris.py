import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.metrics import r2_score
from sklearn.metrics import classification_report
from sklearn.metrics import precision_score

df = pd.read_csv('iris.csv')
print(df)

x = df[['SepalLengthCm','SepalWidthCm','PetalLengthCm']].values
y = df['Species'].values

# training,testing
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.20,random_state=0)
knn = KNeighborsClassifier(n_neighbors=3)
knn.fit(x_train,y_train)
y_pred = knn.predict(x_test)
print(y_pred)
print(y_test)

# Logreg = LogisticRegression()
# # fit the model with data
# Logreg.fit(x_train,y_train)
# # prediction
# y_pred=Logreg.predict(x_test)
# print(y_pred)
# print(y_test)

confussion_metrics=confusion_matrix(y_test,y_pred)
accuracy=accuracy_score(y_test,y_pred)
print("accuracy",accuracy)
# classification_report = classification_report(y_test,y_pred)
# print(classification_report)
precision = precision_score(y_test,y_pred, average='macro')
print("precision",precision)
# r2 = r2_score(y_test,y_pred)
# print(r2)
print("Confusion Metrics :" )
print(confussion_metrics)