import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn import preprocessing
df = pd.read_excel('Book1.xlsx')
# print(df)

le = preprocessing.LabelEncoder()
df['Size'] = le.fit_transform(df['Size'])
# print(df)

x = df[['Weight (grams)','Color Intensity (0-10)','Size']].values
y = df['Fruit Type'].values

from sklearn.preprocessing import StandardScaler

# Standardizing the features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(x)
print(df)

# training,testing
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.20,random_state=0)
knn = KNeighborsClassifier(n_neighbors=3)
knn.fit(x_train,y_train)
df = np.array([[140,7,0]])
y_pred = knn.predict(df)
print(y_pred)
