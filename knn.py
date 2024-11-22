import numpy as np
from sklearn.neighbors import KNeighborsClassifier
weights = [51, 62, 69, 64, 65, 56, 58, 57, 55]
heights = [167, 182, 176, 173, 172, 174, 169, 173, 170]
labels = ['Underweight', 'Normal', 'Normal', 'Normal', 'Normal', 'Underweight', 'Normal', 'Normal', 'Normal']
X = np.array(list(zip(weights, heights)))
Y = np.array(labels)
knn = KNeighborsClassifier(n_neighbors=3)  
knn.fit(X,Y)
test = np.array([[57, 170]])
prediction = knn.predict(test)

print("The class for weight 57 kg and height 170 cm is: " ,prediction)