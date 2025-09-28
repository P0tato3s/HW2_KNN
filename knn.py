#-------------------------------------------------------------------------
# AUTHOR: your name
# FILENAME: title of the source file
# SPECIFICATION: description of the program
# FOR: CS 4210- Assignment #2
# TIME SPENT: how long it took you to complete the assignment
#-----------------------------------------------------------*/

#IMPORTANT NOTE: YOU ARE ALLOWED TO USE ANY PYTHON LIBRARY TO COMPLETE THIS PROGRAM

#Importing some Python libraries
from sklearn.neighbors import KNeighborsClassifier
import pandas as pd
import numpy as np

#Reading the data in a csv file using pandas
db = []
df = pd.read_csv('email_classification.csv')
for _, row in df.iterrows():
    db.append(row.tolist())

#Transform the original features to numbers and add to the 2D array X.
#For instance X = [[f11, f12, ..., f1,20], [f21, f22, ..., f2,20], ...]
#--> add your Python code here
X = df.iloc[:, :-1].astype(float).to_numpy()

#Transform the original classes to numbers and add to the vector Y.
#For instance 'ham' = 0 and 'spam' = 1, Y = [0, 1, 0, 1, ...]
#--> add your Python code here
label_col = df.columns[-1]
label_map = {'ham': 0, 'spam': 1}
Y = df[label_col].map(label_map).to_numpy()

#Implementing the leave-one-out technique to compute the error rate of 1NN
#--> add your Python code here
n = len(Y)
wrong = 0

for i in range(n):
    # leave the i-th instance out
    mask = np.ones(n, dtype=bool)
    mask[i] = False
    X_train, Y_train = X[mask], Y[mask]
    X_test_single = X[i].reshape(1, -1)
    y_true = Y[i]

    # 1-NN with Euclidean distance
    clf = KNeighborsClassifier(n_neighbors=1, metric='euclidean')
    clf.fit(X_train, Y_train)

    #class_predicted = clf.predict([[1, 2, 3, 4, 5, ..., 20]])[0]
    #--> add your Python code here
    class_predicted = clf.predict(X_test_single)[0]

    #Compare the prediction with the true label of the test instance to start calculating the error rate.
    #--> add your Python code here
    if class_predicted != y_true:
        wrong += 1

#Print the error rate
#--> add your Python code here
error_rate = wrong / n if n > 0 else 0.0
print(f'N = {n}')
print(f'Wrong predictions = {wrong}')
print(f'LOO-CV error rate for 1NN = {error_rate:.4f}')
