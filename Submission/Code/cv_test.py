'''
Created on Feb 3, 2017

@author: tapojit
'''
import kaggle
import numpy as np
from sklearn import neighbors, tree, ensemble, naive_bayes, svm, linear_model, kernel_approximation
from sklearn import preprocessing
from sklearn.metrics import accuracy_score
import time
from sklearn.utils import shuffle





#Crossvalidation-Crossvalidation Algorithm
def cv_cv(XL, YL,classifier, split):
    ##Size of each chunk of training data to be passed onto outer loop(each of size 1/split)
    size=abs(len(XL)/split)
    #Validation data of size (1/split)
    val_x, val_y=XL[((split-1)*size):],YL[((split-1)*size):]
    #Training Error
    Error_RF=[]
    Error_DT=[]
    Error_KNN=[]
    diction={'RF_HP1': [],'RF_HP2': [],'RF_HP3': [],'ErrR_HP1': [],'ErrR_HP2': [],'ErrR_HP3': [],
             'DT_HP1': [],'DT_HP2': [], 'ErrD_HP1': [],'ErrD_HP2': [],
             'KNN_HP1': [],'KNN_HP2': [], 'ErrK_HP1': [],'ErrK_HP2': []}
    
    #'RF' is Random forest, 'DT' is Decision tree, 'KNN' is KNN
    if classifier=='RF':   
        for s in range(split-1):
            train_x1, train_y1=XL[(s*size):(size*(1+s))], YL[(s*size):(size*(1+s))]
            #Size of each chunk of training data to be passed onto classifier method(each of size 1/split-1)
            size1=abs(len(train_x1)/(split-1))           
            h,i,j,k,l,m,n=RF(split-1, train_x1, train_y1, size1, val_x, val_y)
            Error_RF.append(h)
            diction['RF_HP1'].append(i)
            diction['RF_HP2'].append(j)
            diction['RF_HP3'].append(k)
            diction['ErrR_HP1'].append(l)
            diction['ErrR_HP2'].append(m)
            diction['ErrR_HP3'].append(n)
        #Determine index with lowest training error. Used to find optimal hyperparameters stored in diction  
        c1=np.argmin(Error_RF)
        cm1=diction['RF_HP1'][c1]
        cm2=diction['RF_HP2'][c1]
        cm3=diction['RF_HP3'][c1]
        diction['ErrR_HP1']=np.mean(diction['ErrR_HP1'],axis=0)
        diction['ErrR_HP2']=np.mean(diction['ErrR_HP2'],axis=0)
        diction['ErrR_HP3']=np.mean(diction['ErrR_HP3'],axis=0)
        finalCLF=ensemble.RandomForestClassifier(n_estimators=cm1,max_features=cm2,min_samples_leaf=cm3)
        finalCLF.fit(XL,YL)
        return finalCLF, 1-np.mean(Error_RF), cm1,cm2,cm3,diction['ErrR_HP1'],diction['ErrR_HP2'],diction['ErrR_HP3']
    
    elif classifier=='DT':   
        for s in range(split-1):
            train_x1, train_y1=XL[(s*size):(size*(1+s))], YL[(s*size):(size*(1+s))]
            #Size of each chunk of training data to be passed onto classifier method(each of size 1/split-1)
            size1=abs(len(train_x1)/(split-1))           
            h,i,j,k,l=DT(split-1, train_x1, train_y1, size1, val_x, val_y)
            Error_DT.append(h)
            diction['DT_HP1'].append(i)
            diction['DT_HP2'].append(j)
            diction['ErrD_HP1'].append(k)
            diction['ErrD_HP2'].append(l)
        #Determine index with lowest training error. Used to find optimal hyperparameters stored in diction  
        c1=np.argmin(Error_DT)
        cm1=diction['DT_HP1'][c1]
        cm2=diction['DT_HP2'][c1]
        diction['ErrD_HP1']=np.mean(diction['ErrD_HP1'],axis=0)
        diction['ErrD_HP2']=np.mean(diction['ErrD_HP2'],axis=0)
        finalCLF=tree.DecisionTreeClassifier(max_depth=cm2,min_samples_leaf=cm1)
        finalCLF.fit(XL,YL)
        return finalCLF, 1-np.mean(Error_DT), cm1,cm2,diction['ErrD_HP1'],diction['ErrD_HP2']

    else:
        for s in range(split-1):
            train_x1, train_y1=XL[(s*size):(size*(1+s))], YL[(s*size):(size*(1+s))]
            #Size of each chunk of training data to be passed onto classifier method(each of size 1/split-1)
            size1=abs(len(train_x1)/(split-1))           
            h,i,j,k,l=KNN(split-1, train_x1, train_y1, size1, val_x, val_y)
            Error_KNN.append(h)
            diction['KNN_HP1'].append(i)
            diction['KNN_HP2'].append(j)
            diction['ErrK_HP1'].append(k)
            diction['ErrK_HP2'].append(l)
        #Determine index with lowest training error. Used to find optimal hyperparameters stored in diction  
        c1=np.argmin(Error_KNN)
        cm1=diction['KNN_HP1'][c1]
        cm2=diction['KNN_HP2'][c1]
        diction['ErrK_HP1']=np.mean(diction['ErrK_HP1'],axis=0)
        diction['ErrK_HP2']=np.mean(diction['ErrK_HP2'],axis=0)
        finalCLF=neighbors.KNeighborsClassifier(n_neighbors=cm2,p=cm1)
        finalCLF.fit(XL,YL)
        return finalCLF, 1-np.mean(Error_KNN), cm1,cm2,diction['ErrK_HP1'],diction['ErrK_HP2']

        

         
        






def RF(split,train_x,train_y, size, valp_x,valp_y):
    error=[]
    n=[]
    #Determining optimum n_estimators value
    for a in range(1,20):
        err=[]
        for i in range(split-1):
            train_x2, train_y2=train_x[(i*size):(size*(1+i))], train_y[(i*size):(size*(1+i))]
            clf=ensemble.RandomForestClassifier(n_estimators=a)
            clf.fit(train_x2,train_y2)
            predicted=clf.predict(train_x[((split-1)*size):])
            err.append(1-accuracy_score(train_y[((split-1)*size):],predicted))
        error.append(np.mean(err))
        n.append(a)
    best_n_ind=np.argmin(np.asarray(error))
    best_n=n[best_n_ind]
    
    error1=[]
    m=[]
    #Determining optimum max_features value
    for a in np.arange(0.1,1.1,0.1):
        err=[]
        for i in range(split-1):
            train_x2, train_y2=train_x[(i*size):(size*(1+i))], train_y[(i*size):(size*(1+i))]
            clf=ensemble.RandomForestClassifier(max_features=a)
            clf.fit(train_x2,train_y2)
            predicted=clf.predict(train_x[((split-1)*size):])
            err.append(1-accuracy_score(train_y[((split-1)*size):],predicted))
        error1.append(np.mean(err))
        m.append(a)
    best_m_ind=np.argmin(np.asarray(error1))
    best_m=m[best_m_ind]
    error2=[]
    s=[]
    #Determining optimum min_samples_leaf value
    for a in range(1,10):
        err=[]
        for i in range(split-1):
            train_x2, train_y2=train_x[(i*size):(size*(1+i))], train_y[(i*size):(size*(1+i))]
            clf=ensemble.RandomForestClassifier(min_samples_leaf=a)
            clf.fit(train_x2,train_y2)
            predicted=clf.predict(train_x[((split-1)*size):])
            err.append(1-accuracy_score(train_y[((split-1)*size):],predicted))
        error2.append(np.mean(err))
        s.append(a)
    best_s_ind=np.argmin(np.asarray(error2))
    best_s=s[best_s_ind]
    finalCLF=ensemble.RandomForestClassifier(n_estimators=best_n,max_features=best_m,min_samples_leaf=best_s)
    finalCLF.fit(train_x,train_y) 
    errorF=1-accuracy_score(valp_y,finalCLF.predict(valp_x))
    return errorF, best_n,best_m,best_s,error,error1,error2

 
         
def DT(split,train_x,train_y,size,valp_x,valp_y):
    error=[]
    min_leaf=[]
    #Determining optimum min_samples_leaf value
    for a in np.arange(0.001,0.0035,0.0005):
        err=[]
        for i in range(split-1):
            train_x1, train_y1=train_x[(i*size):(size*(1+i))], train_y[(i*size):(size*(1+i))]
            clf=tree.DecisionTreeClassifier(min_samples_leaf=a)
            clf.fit(train_x1,train_y1)
            predicted=clf.predict(train_x[((split-1)*size):])
            err.append(1-accuracy_score(train_y[((split-1)*size):],predicted))
        error.append(np.mean(err))
        min_leaf.append(a)
    best_minL_ind=np.argmin(np.asarray(error))
    best_minL=min_leaf[best_minL_ind]
     
    error1=[]
    max_depth=[]
    #Determining optimum max_depth value
    for a in range(1,20):
        err=[]
        for i in range(split-1):
            train_x1, train_y1=train_x[(i*size):(size*(1+i))], train_y[(i*size):(size*(1+i))]
            clf=tree.DecisionTreeClassifier(max_depth=a)
            clf.fit(train_x1,train_y1)
            predicted=clf.predict(train_x[((split-1)*size):])
            err.append(1-accuracy_score(train_y[((split-1)*size):],predicted))
        error1.append(np.mean(err))
        max_depth.append(a)
    best_maxD_ind=np.argmin(np.asarray(error1))
    best_maxD=max_depth[best_maxD_ind]
            
    finalCLF=tree.DecisionTreeClassifier(max_depth=best_maxD,min_samples_leaf=best_minL)
    finalCLF.fit(train_x,train_y)
    errorF=1-accuracy_score(valp_y,finalCLF.predict(valp_x))
    return errorF, best_minL, best_maxD, error, error1

    
            


def KNN(split,train_x,train_y, size, valp_x,valp_y):
    error=[]
    K=[]
    #Determining optimum n_neighbors value
    for a in range(1,10):
        err=[]
        for i in range(split-1):
            train_x2, train_y2=train_x[(i*size):(size*(1+i))], train_y[(i*size):(size*(1+i))]
            clf=neighbors.KNeighborsClassifier(n_neighbors=a)
            clf.fit(train_x2,train_y2)
            predicted=clf.predict(train_x[((split-1)*size):])
            err.append(1-accuracy_score(train_y[((split-1)*size):],predicted))
        error.append(np.mean(err))
        K.append(a)
    best_K_ind=np.argmin(np.asarray(error))
    best_K=K[best_K_ind]
    
    error1=[]
    p=[]
    #Determining optimum p value
    for a in range(1,6):
        err=[]
        for i in range(split-1):
            train_x2, train_y2=train_x[(i*size):(size*(1+i))], train_y[(i*size):(size*(1+i))]
            clf=neighbors.KNeighborsClassifier(p=a)
            clf.fit(train_x2,train_y2)
            predicted=clf.predict(train_x[((split-1)*size):])
            err.append(1-accuracy_score(train_y[((split-1)*size):],predicted))
        error1.append(np.mean(err))
        p.append(a)
    best_p_ind=np.argmin(np.asarray(error1))
    best_p=p[best_p_ind]
    
    finalCLF=neighbors.KNeighborsClassifier(n_neighbors=best_K,p=best_p)
    finalCLF.fit(train_x,train_y)
    errorF=1-accuracy_score(valp_y,finalCLF.predict(valp_x))
    return errorF, best_p, best_K, error, error1



        
            


        
    
