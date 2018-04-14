# Kernel SVM

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from keras.datasets import mnist
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler

# Importing the dataset
(X_train, y_train), (X_test, y_test) = mnist.load_data()

X_train = X_train.reshape(-1,28*28)
X_test = X_test.reshape(-1,28*28)

sc = StandardScaler() 
X_train = sc.fit_transform(X_train)
X_test = sc.fit_transform(X_test)

# Fitting Kernel SVM to the Training set
from sklearn.svm import SVC
classifier = SVC(C=10, kernel = 'linear', random_state = 0)
classifier.fit(X_train, y_train)

y_pred = classifier.predict(X_test)

print('Predicted', len(y_pred), 'digits with accuracy:', accuracy_score(y_test, y_pred))

from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)

with open('resultSVMident.txt', 'w') as f:
   f.write(np.array2string(cm, separator=', '))

sumFRR = 0
sumFAR = 0
for i in range(10):
    FRR = (cm[i,:].sum() - cm[i][i])/cm[i,:].sum()
    FAR = (cm[:,i].sum() - cm[i][i])/(cm[:,:].sum() - cm[i,:].sum())
    
    with open('./svmIdentaccuracy.txt', "a") as myfile:
        myfile.write("User: " + str(i) + "\nFAR: " + str("%.5f" % FAR) + "\nFRR: " + str("%.5f" % FRR) + "\n\n\n") 
    
    sumFRR = sumFRR + FRR
    sumFAR = sumFAR + FAR                                                        


with open('./svmIdentaccuracy.txt', "a") as myfile:
        myfile.write("Mean: \nFAR: " + str("%.5f" % (sumFAR/10)) + "\nFRR: " + str("%.5f" % (sumFRR/10)) + "\n\n\n")

#if (best_parameters.kernel == 'rbf'):
#    classifier_best = SVC(kernel = best_parameters.kernel, C  = best_parameters.C, gamma = best_parameters.gamma, random_state = 0)
#else:
#    classifier_best = SVC(kernel = best_parameters.kernel, C  = best_parameters.C, random_state = 0)
#
#classifier_best.fit(X_train, y_train)
#
## Predicting the Test set results
#y_pred = classifier_best.predict(X_test)
#
#print('Predicted', len(y_pred), 'digits with accuracy:', accuracy_score(y_test, y_pred))
#
## Making the Confusion Matrix
#from sklearn.metrics import confusion_matrix
#cm = confusion_matrix(y_test, y_pred)
#
#with open('resultSVMident.txt', 'w') as f:
#    f.write(np.array2string(cm, separator=', '))