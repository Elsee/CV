import os
from sklearn.model_selection import train_test_split  
from sklearn import svm
import numpy as np
import pandas
from time import gmtime, strftime
import csv

sumFRR = 0;
sumFAR = 0;

for segment in [1,2,4,7,8]:
    row_accuracy = []
    for num in range (10):
        filenames = os.listdir('resultsDenoisingSparse/segments' + str(segment))    
        
        target_filepath ='AEResult_#' + str(num) + '.csv'
        
        target_number = pandas.read_csv('resultsDenoisingSparse/segments' + str(segment) +'/'+target_filepath, header = None)
        target_number['target'] = 1
        target_labels = target_number['target']
        target_number.drop(["target"], axis=1, inplace=True)
        
        outliers = pandas.DataFrame(columns=range(target_number.shape[1]))
        
        for item in filenames:
            # Load current dataset
            url = item
            if (url != target_filepath):
                dataset = pandas.read_csv('resultsDenoisingSparse/segments' + str(segment) +'/'+url, header = None, engine='python')
                outliers = np.r_[outliers, dataset]
                
        outliers = pandas.DataFrame(outliers)
        outliers["number"] = -1
        neg_labels = outliers["number"]
        outliers.drop(["number"], axis=1, inplace=True)
        
        train_data, test_data, train_target, test_target = train_test_split(target_number, target_labels, train_size = 0.7, test_size = 0.3)
        
        model = svm.OneClassSVM(kernel = 'linear', nu=0.1)
        test_data_with_outliers = np.r_[test_data, outliers]
        y_score = model.fit(train_data).decision_function(test_data_with_outliers) 
        
        y_pred_train =  model.predict(train_data) 
        y_pred_test = model.predict(test_data)
        
        from sklearn.metrics import confusion_matrix
        cm = confusion_matrix(test_target, y_pred_test)
        
        y_pred_outliers = model.predict(outliers)
        
        cm1 = confusion_matrix(neg_labels, y_pred_outliers)
        if (cm1.shape[1] != 2):
            cm1 = np.append(cm1, [[0]], axis=1)
        if (cm1.shape[0] != 2):
            cm1 = np.append(cm1, [[0,0]], axis=0)
        if (cm.shape[1] != 2):
            cm = np.append([[0]], cm, axis=1)
        if (cm.shape[0] != 2):
            cm = np.append([[0,0]], cm, axis=0)
        totalCM = cm + cm1
        totalCM = cm + cm1
        
        with open('svmOneClassWithDenoisingAE/segments' + str(segment) +'/' + 'resultSVMauthDenoisingSparse' + str(num) +'.txt', 'w') as f:
           f.write(np.array2string(totalCM, separator=', '))
         
        FRR = totalCM[1][0] / (totalCM[1][0] + totalCM[1][1])
        FAR = totalCM[0][1] / (totalCM[0][0] + totalCM[0][1])
        
        sumFRR = sumFRR + FRR
        sumFAR = sumFAR + FAR
        
        row_accuracy.append(str("%.2f" % ((totalCM[1][1]+totalCM[0][0])/(totalCM[1][1]+totalCM[0][0]+totalCM[0][1]+totalCM[1][0]))))

           
        with open('svmOneClassWithDenoisingAE/segments' + str(segment) +'/' + 'resultSVMauthAccuracy.txt', 'a') as f:
           f.write("User: " + str(num) + "\nFAR: " + str("%.5f" % FAR) + "\nFRR: " + str("%.5f" % FRR) + "\n\n\n")
    
    with open('results_new_accuracy/segments' + str(segment) +'/' + 'output.csv','a') as f:
            writer = csv.writer(f, dialect='excel')
            writer.writerow(row_accuracy)
          
    with open('svmOneClassWithDenoisingAE/segments' + str(segment) +'/' + 'resultSVMauthAccuracy.txt', "a") as myfile:
        myfile.write("Mean: \nFAR: " + str("%.5f" % (sumFAR/10)) + "\nFRR: " + str("%.5f" % (sumFRR/10)) + "\n\n\n")