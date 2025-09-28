#-------------------------------------------------------------------------
# AUTHOR: John Li
# FILENAME: knn.py
# SPECIFICATION: Using KNN to classify emails as spam or not spam
# FOR: CS 4210- Assignment #2
# TIME SPENT: how long it took you to complete the assignment
#-----------------------------------------------------------*/

#IMPORTANT NOTE: YOU ARE ALLOWED TO USE ANY PYTHON LIBRARY TO COMPLETE THIS PROGRAM

#Importing some Python libraries
from sklearn.neighbors import KNeighborsClassifier
import numpy as np
import pandas as pd

#Reading the data in a csv file using pandas
db = []
df = pd.read_csv('email_classification.csv')
for _, row in df.iterrows():
    db.append(row.tolist())

#Loop your data to allow each instance to be your test set
for i in db:

    #Add the training features to the 20D array X removing the instance that will be used for testing in this iteration.
    #For instance, X = [[1, 2, 3, 4, 5, ..., 20]].
    #Convert each feature value to float to avoid warning messages
    X = df.iloc[:, :-1].astype(float).to_numpy()

    #Transform the original training classes to numbers and add them to the vector Y.
    #Do not forget to remove the instance that will be used for testing in this iteration.
    #For instance, Y = [1, 2, ,...].
    #Convert each feature value to float to avoid warning messages
    Y = df.iloc[:, -1].to_numpy()

    #Fitting the knn to the data using k = 1 and Euclidean distance (L2 norm)
    #--> add your Python code here
    clf = KNeighborsClassifier(n_neighbors=2, metric='euclidean')
    clf.fit(X, Y)

    nn_idx = clf.kneighbors(X, return_distance=False)

    loo_idx = nn_idx[:, 1]
    Y_pred = Y[loo_idx]

    #Compare the prediction with the true label of the test instance to start calculating the error rate.
    wrong = int(np.sum(Y_pred != Y))
    N = len(Y)
    error_rate = wrong / N

#Print the error rate
print(f"N = {N}")
print(f"Wrong predictions = {wrong}")
print(f"LOO-CV error rate for 1NN = {error_rate:.4f}")






