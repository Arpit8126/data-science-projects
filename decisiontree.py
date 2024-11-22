import pandas as pd
df = pd.read_csv('diabetes.csv')
df.dropna(inplace=True)
X1 = df.columns[:-1].tolist() 
Y1 = df.columns[-1]
X = df[[X1]].values
Y = df[Y1].values
from sklearn import preprocessing
le = preprocessing.LabelEncoder()
categorical_columns = df.select_dtypes(include=['object']).columns
for columnS in categorical_columns:
    df[columnS] = le.fit_transform(df[column])
from sklearn.model_selection import train_test_split
X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=0.20,random_state=0)
from sklearn.tree import DecisionTreeClassifier
model = DecisionTreeClassifier(max_depth=3, criterion='entropy')
model.fit(X_train, Y_train)
y_pred = model.predict(X_test)
from sklearn.tree import plot_tree
import matplotlib.pyplot as plt
plt.figure(figsize=(15, 10))
plot_tree(model, feature_names=X.columns, class_names=['NonDiabetic', 'Diabetic'], filled=True)
plt.show()

