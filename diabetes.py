import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
df = pd.read_csv('diabetes.csv')
# print(df)
# print(df.head())
X = df[['Pregnancies','Glucose','BloodPressure','SkinThickness','Insulin','BMI','DiabetesPedigreeFunction','Age']].values
Y = df['Outcome'].values
print(X)
print(Y)

X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=0.20,random_state=1)

# instentiate model
logreg = LogisticRegression()

# fil model
logreg.fit(X_train,Y_train)

# prediction of test data
Y_pred=logreg.predict(X_test)

print(Y_pred)
print(Y_test)
