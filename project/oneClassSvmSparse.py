import glob
from sklearn.model_selection import train_test_split  
from sklearn import svm
import numpy as np
import pandas
from time import gmtime, strftime
import os

for num in range (10):
    filenames = os.listdir('resultsSparse')
    
    target_filepath ='AEResult_#' + str(num) + '.csv'
    
    outliers = pandas.DataFrame(columns=range(32))
    
    for item in filenames:
        # Load current dataset
        url = item
        if (url != target_filepath):
            dataset = pandas.read_csv('resultsSparse/'+url, header = None, engine='python')
            outliers = np.r_[outliers, dataset]
            
    outliers = pandas.DataFrame(outliers)
    outliers["number"] = -1
    neg_labels = outliers["number"]
    outliers.drop(["number"], axis=1, inplace=True)
    
    target_number = pandas.read_csv('resultsSparse/'+target_filepath, header = None)
    target_number['target'] = 1
    
    target_labels = target_number['target']
    
    target_number.drop(["target"], axis=1, inplace=True)
    
    train_data, test_data, train_target, test_target = train_test_split(target_number, target_labels, train_size = 0.8, test_size = 0.2)
    
    model = svm.OneClassSVM(kernel = 'linear', nu=0.9)
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
    
    with open('svmOneClassWithAE/resultSVMauthSparse' + str(num) +'.txt', 'w') as f:
       f.write(np.array2string(totalCM, separator=', '))