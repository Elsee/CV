from keras.datasets import cifar100
import numpy as np
import csv

(x_train, y_train), (x_test, y_test) = cifar100.load_data()

x_train = np.reshape(x_train, (x_train.shape[0], -1))
x_test = np.reshape(x_test, (x_test.shape[0], -1))

sumFRR = 0;
sumFAR = 0;
row_accuracy = []
        
for img in range (100):
    target_class_labels = np.where([y_train[:]==img])[1]
    target_class_data = x_train[:][target_class_labels]
    y_test_temp = np.copy(y_test)
    
    for i in range(len(y_test_temp)):
        if y_test_temp[i] == img:
            y_test_temp[i] = 1
        else:
            y_test_temp[i] = -1
    
    test_target_idxs = np.where([y_test_temp[:]==1])[1]        
    test_target_class_data = x_test[:][test_target_idxs]
    test_target_labels = y_test_temp[y_test_temp[:]==1]
    
    outliers_idxs = np.where([y_test_temp[:]!=1])[1] 
    outliers_class_data = x_test[:][outliers_idxs]
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
    myfile.write("Mean: \nFAR: " + str("%.5f" % (sumFAR/100)) + "\nFRR: " + str("%.5f" % (sumFRR/100)) + "\n\n\n")