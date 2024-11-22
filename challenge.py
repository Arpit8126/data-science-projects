import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier

weights = [150, 130,180,100, 120, 110,115, 105, 160, 150, 155, 140]
Color_Intensity = [7, 6, 8, 5, 3, 4, 3, 2, 9,8,9,7]
size = ['large', 'medium', 'large', 'small', 'small', 'medium', 'medium', 'small', 'large','medium','large','medium']
labels = ['apple', 'apple', 'apple', 'apple', 'banana', 'banana', 'banana', 'banana', 'orange', 'orange', 'orange','orange']

size_mapping = {'small': 1, 'medium': 2, 'large': 3}
size = [size_mapping[s] for s in size]

X = np.array(list(zip(weights, Color_Intensity, size)))
Y = np.array(labels)

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)



X_train, X_test, Y_train, Y_test = train_test_split(X_scaled, Y, test_size=0.2, random_state=0)

knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train, Y_train)

test = np.array([[140, 7, 2]])
prediction = knn.predict(test)
print(prediction)