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

sumFRR = 0;
sumFAR = 0;
        
for num in range (10):
    target_class_labels = y_train[y_train[:]==num]
    target_class_data = X_train[:][target_class_labels]
    y_test_temp = np.copy(y_test)
    
    for i in range(len(y_test_temp)):
        if y_test_temp[i] == num:
            y_test_temp[i] = 1
        else:
            y_test_temp[i] = -1
            
    test_target_class_data = X_test[y_test_temp[:]==1]
    test_target_labels = y_test_temp[y_test_temp[:]==1]
    
    outliers_class_data = X_test[y_test_temp[:]!=1]
    outliers_labels = y_test_temp[y_test_temp[:]==-1]
                                  
    model = svm.OneClassSVM(nu=0.1,kernel='linear')
    model.fit( target_class_data )
    
    from sklearn.metrics import confusion_matrix
    y_pred_train = model.predict(target_class_data)
    y_pred_test = model.predict(test_target_class_data)
    y_pred_outliers = model.predict(outliers_class_data)
    
    cm = confusion_matrix(test_target_labels, y_pred_test)
    cm1 = confusion_matrix(outliers_labels, y_pred_outliers)
    
    if (cm1.shape[1] != 2):
        cm1 = np.append(cm1, [[0]], axis=1)
    if (cm1.shape[0] != 2):
        cm1 = np.append(cm1, [[0,0]], axis=0)
    if (cm.shape[1] != 2):
        cm = np.append([[0]], cm, axis=1)
    if (cm.shape[0] != 2):
        cm = np.append([[0,0]], cm, axis=0)
    total_cm = cm + cm1
    
    FRR = total_cm[1][0] / (total_cm[1][0] + total_cm[1][1])
    FAR = total_cm[0][1] / (total_cm[0][0] + total_cm[0][1])
    
    sumFRR = sumFRR + FRR
    sumFAR = sumFAR + FAR
    
    with open('svmOneClassWithoutAE/resultSVMauth#'+str(num)+'.txt', 'w') as f:
       f.write(np.array2string(total_cm, separator=', '))
       
    with open('svmOneClassWithoutAE/resultSVMauthAccuracy.txt', 'a') as f:
       f.write("User: " + str(num) + "\nFAR: " + str("%.5f" % FAR) + "\nFRR: " + str("%.5f" % FRR) + "\n\n\n")
      
with open('svmOneClassWithoutAE/resultSVMauthAccuracy.txt', "a") as myfile:
    myfile.write("Mean: \nFAR: " + str("%.5f" % (sumFAR/10)) + "\nFRR: " + str("%.5f" % (sumFRR/10)) + "\n\n\n")