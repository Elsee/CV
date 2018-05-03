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
for i in range(1,6):
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

classes = ['plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
num_classes = len(classes)
samples_per_class = 7
for y, cls in enumerate(classes):
    idxs = np.flatnonzero(y_train == y)
    idxs = np.random.choice(idxs, samples_per_class, replace=False)
    for i, idx in enumerate(idxs):
        plt_idx = i * num_classes + y + 1
        plt.subplot(samples_per_class, num_classes, plt_idx)
        plt.imshow(X_train[idx].astype('uint8'))
        plt.axis('off')
        if i == 0:
            plt.title(cls)
plt.show()

X_train = np.reshape(X_train, (X_train.shape[0], -1))
X_test = np.reshape(X_test, (X_test.shape[0], -1))

sc = StandardScaler() 
X_train = sc.fit_transform(X_train)
X_test = sc.fit_transform(X_test)

# Fitting Kernel SVM to the Training set
from sklearn.svm import SVC
classifier = SVC(C=10, kernel='rbf', gamma=0.001)

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