from sklearn.datasets import fetch_mldata
from sklearn import svm
from sklearn.cross_validation import train_test_split
import numpy as np
from collections import Counter
from keras.datasets import mnist
import gzip
from sklearn.preprocessing import StandardScaler

(X_train, y_train), (X_test, y_test) = mnist.load_data()
 
X_train = X_train.reshape(-1,28*28)
X_test = X_test.reshape(-1,28*28)

sc = StandardScaler() 
X_train = sc.fit_transform(X_train)
X_test = sc.fit_transform(X_test)
y_train = y_train.astype('int')
y_test = y_test.astype('int')

# build a one class svm model

target_class_labels = y_train[y_train[:]==0]
target_class_data = X_train[:][target_class_labels] 

for i in range(len(y_test)):
    if y_test[i] == 0:
        y_test[i] = 1
    else:
        y_test[i] = -1
        
test_target_class_data = X_test[y_test[:]==1]
test_target_labels = y_test[y_test[:]==1]

outliers_class_data = X_test[y_test[:]!=1]
outliers_labels = y_test[y_test[:]==-1]
                              
model = svm.OneClassSVM(kernel='rbf', random_state = 0)  
model.fit( target_class_data )

from sklearn.model_selection import cross_val_score
accuracies = cross_val_score(estimator = model, X = target_class_data, y = target_class_labels, cv = 10, scoring="accuracy")
accuracies.mean()
accuracies.std()

from sklearn.model_selection import GridSearchCV
parameters = [{'nu': [0.1, 0.5, 0.6, 0.7, 0.8, 0.9], 'kernel': ['linear']},
              {'nu': [0.1, 0.5, 0.6, 0.7, 0.8, 0.9], 'kernel': ['rbf'], 'gamma': [0.1, 0.5, 0.9, 'auto']}]
grid_search = GridSearchCV(estimator = model,
                           param_grid = parameters,
                           scoring = 'accuracy',
                           cv = 10)
grid_search = grid_search.fit(target_class_data, target_class_labels)
best_accuracy = grid_search.best_score_
best_parameters = grid_search.best_params_

with open('./gridSearchSVMresultsOneClass.txt','a') as f:
    f.write('Best parameters: ' + str(best_parameters) + '\n')