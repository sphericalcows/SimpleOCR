# -*- coding: utf-8 -*-
"""
Created on Tue Jul 22 08:58:01 2014

@author: Freeman

This code is based on the sample code in the SKLearn tutorials, written by
Gael Varoquaux <gael dot varoquaux at normalesup dot org> and licensed under
BSD 3 clause.

The program reads in the files TrainingData.txt and TestData.txt, both
using pickle. 
These files are produced using the ExtractNumericData program and encode 
the numeric value and an image containing the character.
    data set is array of [value, image] 
    (I am sure that this should be done as a class, but this is pretty
     quick and dirty at the moment)
"""

# Standard scientific Python imports
#import pylab as pl

# Import classifiers and performance metrics
from sklearn import metrics
#from sklearn.cross_validation import train_test_split
#from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.lda import LDA
from sklearn.qda import QDA


# Other dependencies
import pickle

# The digits dataset
infile = open("C:\Users\Freeman\Documents\OMR\TrainingData.txt", "r")
templateSet=[]
templateSet = pickle.load(infile)
infile.close()
n_samples = len(templateSet)

#The test dataset
infile = open("C:\Users\Freeman\Documents\OMR\TestData.txt", "r")
testSet=[]
testSet=pickle.load(infile)
infile.close()
n_test = len(testSet)

# The data from the forms is made of 5x3 images of digits.
# For these we images know which digit they represent: 
# it is given in the first entry of the dataset, with the image following.

# To apply an classifier on this data, we need to flatten the image:
# We will also allow for filtering out a variant using excludeSet
excludeSet=[16,17,18,19,20]
#excludeSet =[]

sampleMat = [templateSet[i][1] for i in range(0,n_samples) if i%21 not in excludeSet]
data = [sampleMat[i].reshape(-1) for i in range(0,len(sampleMat))]
dataVal = [templateSet[i][0] for i in range(0,n_samples)if i%21 not in excludeSet]

dataMat = [testSet[i][1] for i in range(0,n_test) if i%21 not in excludeSet]
testdata = [dataMat[i].reshape(-1) for i in range(0,len(dataMat))]
testVal = [testSet[i][0] for i in range(0,n_test) if i%21 not in excludeSet]

# Test Classifiers. This code modified from skLearn documentation.
names = ["Nearest Neighbors", "Linear SVM", "RBF SVM", "Decision Tree",
         "Random Forest", "AdaBoost", "Naive Bayes", "LDA", "QDA"]
classifiers = [
    KNeighborsClassifier(3),
    SVC(kernel="linear", C=0.025),
    SVC(gamma=2, C=1),
    DecisionTreeClassifier(max_depth=5),
    RandomForestClassifier(max_depth=5, n_estimators=10, max_features=1),
    AdaBoostClassifier(),
    GaussianNB(),
    LDA(),
    QDA()]

#Try different classifiers with these data sets
for name, clf in zip(names, classifiers):
    # We learn the digits on the template digits
    clf.fit(data, dataVal)

    # Now predict the value of the test digits:
    expected = testVal
    predicted = clf.predict(testdata)

    print("Classification report for classifier %s:\n%s\n" % (clf, metrics.classification_report(expected, predicted)))
    print("Confusion matrix:\n%s" % metrics.confusion_matrix(expected, predicted))
    pause = raw_input("Press enter to continue (enter 'exit' to end)")
    if pause == 'exit':
        break
    print()
    
print "End of test run"
