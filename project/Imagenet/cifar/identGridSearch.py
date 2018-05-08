# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler

def unpickle(file):
    import pickle
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict

X_train = np.array([])
X_train.shape=(0, 32, 32, 3)

y_train = np.array([])

# Importing the dataset
for i in range(1,2):
    data = unpickle('../cifar-10-batches-py/data_batch_' + str(i))
    dataset = data[b'data']
    labels = data[b'labels']
    dataset = dataset.reshape(10000, 3, 32, 32).transpose(0,2,3,1).astype("float")
    labels = np.array(labels)
    X_train = np.concatenate( (X_train,dataset) )
    y_train = np.concatenate( (y_train,labels) )

test_data = unpickle('../cifar-10-batches-py/test_batch')
X_test = test_data[b'data']
X_test = X_test.reshape(10000, 3, 32, 32).transpose(0,2,3,1).astype("float")
y_test = test_data[b'labels']
y_test = np.array(y_test)

X_train = np.reshape(X_train, (X_train.shape[0], -1))
X_test = np.reshape(X_test, (X_test.shape[0], -1))

sc = StandardScaler() 
X_train = sc.fit_transform(X_train)

# Fitting Kernel SVM to the Training set
from sklearn.svm import SVC

from sklearn.grid_search import GridSearchCV

parameter_candidates = [
  {'C': [1, 10, 100, 1000], 'kernel': ['linear']},
  {'C': [1, 10, 100, 1000], 'gamma': [0.001, 0.0001], 'kernel': ['rbf']},
]

clf = GridSearchCV(estimator=SVC(), param_grid=parameter_candidates, n_jobs=-1)

# Train the classifier on training data
clf.fit(X_train, y_train)

# Print out the results 
print('Best score for training data:', clf.best_score_)
print('Best `C`:',clf.best_estimator_.C)
print('Best kernel:',clf.best_estimator_.kernel)
print('Best `gamma`:',clf.best_estimator_.gamma)