import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.datasets import make_classification
from sklearn.ensemble import BaggingClassifier, VotingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, precision_score, recall_score, f1_score

df = pd.read_csv('Iris.csv')
X1 = df.columns[:-1].tolist()  
Y1 = df.columns[-1]           
X=df[X1].values
Y=df[Y1].values
#X,Y=make_classification(n_samples=1000,n_features=10,random_state=20)
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=20)
decision_tree = DecisionTreeClassifier(random_state=20)
knn = KNeighborsClassifier(n_neighbors=5)
logistic = LogisticRegression(random_state=20)
voting = VotingClassifier(estimators=[
    ('knn', knn),
    ('log_reg', logistic),
    ('dtree', decision_tree)
], voting='hard') 



bagging_model = BaggingClassifier(estimator=voting, n_estimators=15, random_state=20)
bagging_model.fit(X_train, Y_train)
# base_model = DecisionTreeClassifier(max_depth=3, criterion='entropy')

# base_model.fit(X_train, Y_train)
# Y_pred = base_model.predict(X_test)
Y_pred = bagging_model.predict(X_test)
accuracy = accuracy_score(Y_test, Y_pred)
print(f"Accuracy of the Bagging model: {accuracy}")
