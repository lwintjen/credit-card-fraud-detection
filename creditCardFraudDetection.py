"""
Author : Loris Wintjens
Goal : Detect credit card fraud on 280.000 transactions data set
"""
# import the necessary packages
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, accuracy_score
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor

# load the dataset from the csv using pandas
data = pd.read_csv("creditcard.csv")

# explore the data set
data = data.sample(frac = 0.1, random_state=1) # cut the dataset otherwise it'd take too much time to run 


# determine number of fraud cases in dataset
fraud = data[data['Class'] == 1]
valid = data[data['Class'] == 0]

outlier_fraction = len(fraud) /float(len(valid))


# get all the columns from the data set
columns = data.columns.tolist()

# filter the columns to remove data we do not want
columns = [ c for c in columns if c not in ["Class"]]

# store the variable we'll be predicting on
target = 'Class'

X = data[columns]
Y = data[target]

# define a random state
state = 1

# define the outlier detection methods
classifiers = {
    "Isolation Forest": IsolationForest(max_samples=len(X),
                                        contamination=outlier_fraction,
                                        random_state=state),
    "Local Outlier Factor" : LocalOutlierFactor(n_neighbors= 20, 
                                                contamination=outlier_fraction)
}
# fit the model
numberOutliers = len(fraud)

for i,(clf_name, clf) in enumerate(classifiers.items()):
    #fit the data and tag outliers
    if clf_name == "Local Outlier Factor":
        y_pred = clf.fit_predict(X)
        score_pred = clf.negative_outlier_factor_
    else :
        clf.fit(X)
        scores_pred = clf.decision_function(X)
        y_pred = clf.predict(X)
    
    # reshape the prediction values to 0 for valid, 1 for fraud
    y_pred[y_pred == 1] = 0
    y_pred[y_pred == - 1] = 1

    nErrors = (y_pred != Y).sum()

    # run the classification metrics
    print('{}: {}'.format(clf_name, nErrors))
    print(accuracy_score(Y, y_pred))
    print(classification_report(Y, y_pred))