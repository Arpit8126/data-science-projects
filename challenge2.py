import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
df=pd.read_csv('iris.data.csv')
X=df[['sepalL','sepalW','petalL','petalW']].values
Y=df['target'].values
print(X,Y)
X = np.array(list(zip(df['sepalL'],df['sepalW'],df['petalL'], df['petalW'])))
Y = np.array(df['target'])
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
X_train, X_test, Y_train, Y_test = train_test_split(X_scaled, Y, test_size=0.2, random_state=0)
knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train, Y_train)
prediction = knn.predict(X_test)
print(prediction)
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from sklearn.metrics import precision_score
confussion_metrics=confusion_matrix(Y_test,prediction)
accuracy=accuracy_score(Y_test,prediction)
classification_report=classification_report(Y_test,prediction)
print("Confusion Metrics :" )
print(confussion_metrics)
print("Accuracy :")
print(accuracy)
print("Classification Report :")
print(classification_report)
