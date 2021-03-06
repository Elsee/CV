# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import csv

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

# build a one class svm model

sumFRR = 0;
sumFAR = 0;
row_accuracy = []
        
for img in range (10):
    target_class_labels = np.where([y_train[:]==img])[1]
    target_class_data = X_train[:][target_class_labels]
    y_test_temp = np.copy(y_test)
    
    for i in range(len(y_test_temp)):
        if y_test_temp[i] == img:
            y_test_temp[i] = 1
        else:
            y_test_temp[i] = -1
            
    test_target_idxs = np.where([y_test_temp[:]==1])[1]        
    test_target_class_data = X_test[:][test_target_idxs]
    test_target_labels = y_test_temp[y_test_temp[:]==1]
    
    outliers_idxs = np.where([y_test_temp[:]!=1])[1] 
    outliers_class_data = X_test[:][outliers_idxs]
    outliers_labels = y_test_temp[y_test_temp[:]==-1]
     
    # Fitting Kernel SVM to the Training set
    from sklearn import svm                            
    model = svm.OneClassSVM(nu=0.1,kernel='linear', gamma=0.001)
    model.fit( target_class_data )
    
    from sklearn.metrics import confusion_matrix
    y_pred_train = model.predict(target_class_data)
    y_pred_test = model.predict(test_target_class_data)
    y_pred_outliers = model.predict(outliers_class_data)
    
    cm = confusion_matrix(test_target_labels, y_pred_test)
    cm1 = confusion_matrix(outliers_labels, y_pred_outliers)
    
    if(cm1.shape[0] != 2):
        cm1_extended = np.zeros((2,2))
        cm1_extended[0][0] = cm1
        total_cm = cm + cm1_extended
    else:
        total_cm = cm + cm1
    
    FRR = total_cm[1][0] / (total_cm[1][0] + total_cm[1][1])
    FAR = total_cm[0][1] / (total_cm[0][0] + total_cm[0][1])
    
    sumFRR = sumFRR + FRR
    sumFAR = sumFAR + FAR
    
    row_accuracy.append(str("%.2f" % ((total_cm[1][1]+total_cm[0][0])/(total_cm[1][1]+total_cm[0][0]+total_cm[0][1]+total_cm[1][0]))))

    with open('svmOneClassWithoutAE/resultSVMauth#'+str(img)+'.txt', 'w') as f:
       f.write(np.array2string(total_cm, separator=', '))
       
    with open('svmOneClassWithoutAE/resultSVMauthAccuracy.txt', 'a') as f:
       f.write("User: " + str(img) + "\nFAR: " + str("%.5f" % FAR) + "\nFRR: " + str("%.5f" % FRR) + "\n\n\n")
       
with open('results_new_accuracy/output.csv','a') as f:
    writer = csv.writer(f, dialect='excel')
    writer.writerow(row_accuracy)
      
with open('svmOneClassWithoutAE/resultSVMauthAccuracy.txt', "a") as myfile:
    myfile.write("Mean: \nFAR: " + str("%.5f" % (sumFAR/10)) + "\nFRR: " + str("%.5f" % (sumFRR/10)) + "\n\n\n")