# run_me.py module

import matplotlib.pyplot as plt
import kaggle

# Assuming you are running run_me.py from the Submission/Code directory, otherwise the path variable will be different for you
import numpy as np
from sklearn import neighbors, tree, ensemble, svm
from sklearn import preprocessing
from sklearn.metrics import accuracy_score
import time
from sklearn.feature_selection import VarianceThreshold
from sklearn.utils import shuffle
from cv_test import cv_cv
# Load the Email spam data
path = '../../Data/Email_spam/'
train = np.load(path + 'train.npy')
test = np.load(path + 'test_distribute.npy')

# Load the Occupancy_detection data
path1 = '../../Data/Occupancy_detection/'
train1 = np.load(path1 + 'train.npy')
test1 = np.load(path1 + 'test_distribute.npy')
# Load the USPS_digits data
path2 = '../../Data/USPS_digits/'
train2 = np.load(path2 + 'train.npy')
test2 = np.load(path2 + 'test_distribute.npy')

 
#Plots line graph
def plot_line(x,y,x_lab):
    inds=x
    values=y
    title='Error vs. '+x_lab
    #Plot a line graph
    plt.figure(2, figsize=(6,4))  #6x4 is the aspect ratio for the plot
    plt.plot(inds,values,'or-', linewidth=3) #Plot the first series in red with circle marker
    
    #This plots the data
    plt.grid(True) #Turn the grid on
    plt.ylabel("Training_Error") #Y-axis label
    plt.xlabel(x_lab) #X-axis label
    plt.title(title) #Plot title
    plt.xlim(0,max(inds)) #set x axis range
    plt.ylim(0,1) #Set yaxis range
    
    #Make sure labels and titles are inside plot area
    plt.tight_layout()
    
    #Save the chart
    plt.savefig("../Figures/"+title+".png")
    print "Line graph images generated"

def trainer(model,feature_selection1,train_data,test_data, data_set,split):
    #features with low variance removed using formula Var(x)=p(1-p). Here, the equation is looking for features with variance below 80%
    if feature_selection1:    
        X_all=np.concatenate((train_data[:,1:],test_data[:,1:]),axis=0)
        sel = VarianceThreshold(threshold=(.8 * (1 - .8)))
        X_red=sel.fit_transform(X_all)
        X_learn=X_red[:(len(X_red)/2)]
        X_test=X_red[(len(X_red)/2):]
        Y_learn=train_data[:,0]
        X_learnS, Y_learnS=shuffle(X_learn,Y_learn)
    else:
        X_learnS, Y_learnS=shuffle(train_data[:,1:],train_data[:,0])
        X_test=test_data[:,1:]
    #best_model: trained model using optimized hyperparameters; errorF: Mean training error; c1: optimized n_estimators;     
    #c2: optimized max_features; c3: optimized min_samples_leaf; e1: array of training errors for n_estimators; e2: array of training errors for max_features
    #e3: array of training errors for min_samples_leaf
    if model=='RF':
        best_model, errorF, c1, c2, c3,e1,e2,e3=cv_cv(X_learnS, Y_learnS, model, split)
        print model,"for", data_set, "- Mean Training Error, optimized n_estimators,max_features,min_samples_leaf: ", errorF, c1, c2,c3
        plot_line(range(1,20),e1,'n_estimators')
        plot_line(np.arange(0.1,1.1,0.1),e2,'max_features')
        plot_line(range(1,10),e3,'min_samples_leaf')
    elif model=='DT':
        best_model, errorF, c1, c2,e1,e2=cv_cv(X_learnS, Y_learnS, model, split)
        print model,"for", data_set, "- Mean Training Error, optimized min_samples_leaf, max_depth: ", errorF, c1, c2
    else: 
        best_model, errorF, c1, c2,e1,e2=cv_cv(X_learnS, Y_learnS, model, split)
        print model,"for", data_set, "- Mean Training Error, optimized p, n_neighbors: ", errorF, c1, c2

    reel=best_model.predict(X_test)
    # 
    # #Save prediction file in Kaggle format
    predictions = reel
    kaggle.kaggleize(predictions, "../Predictions/"+data_set+"/test.csv")
trainer('RF', False, train, test, 'Email_spam', 4)

#Uncomment and run other classifiers
# trainer('DT', False, train1, test1, 'Occupancy_detection', 5)
# trainer('KNN', False, train2, test2, 'USPS_digits', 6)








#Using default hyperparameters for all classifiers
def default_knn(test_x,train_x,train_y):
    clf1a=neighbors.KNeighborsClassifier()
    time_sA_tr=time.time()
    clf1a.fit(train_x,train_y)
    time_eA_tr=time.time()
    #time taken to train
    durA_tr=time_eA_tr-time_sA_tr
    time_sA_te=time.time()
    clf1a.predict(test_x)
    time_eA_te=time.time()
    #time taken to test
    durA_te=time_eA_te-time_sA_te
    #Accuracy on train data(model is retrained after a split of 50%/50%)
    length=len(train_x)/2
    clf1a_te=ensemble.RandomForestClassifier()
    clf1a_te.fit(train_x[:length],train_y[:length])
    acc1A=accuracy_score(train_y[length:],clf1a_te.predict(train_x[length:]))
    return acc1A,durA_tr,durA_te
#Decision tree
def default_DT(test_x,train_x,train_y):
    clf1a=tree.DecisionTreeClassifier()
    time_sA_tr=time.time()
    clf1a.fit(train_x,train_y)
    time_eA_tr=time.time()
    #time taken to train
    durA_tr=time_eA_tr-time_sA_tr
    time_sA_te=time.time()
    clf1a.predict(test_x)
    time_eA_te=time.time()
    #time taken to test
    durA_te=time_eA_te-time_sA_te
    #Accuracy on train data(model is retrained after a split of 50%/50%)
    length=len(train_x)/2
    clf1a_te=ensemble.RandomForestClassifier()
    clf1a_te.fit(train_x[:length],train_y[:length])
    acc1A=accuracy_score(train_y[length:],clf1a_te.predict(train_x[length:]))
    return acc1A,durA_tr,durA_te
#Random Forest
def default_RF(test_x,train_x,train_y):
    clf1a=ensemble.RandomForestClassifier()
    time_sA_tr=time.time()
    clf1a.fit(train_x,train_y)
    time_eA_tr=time.time()
    #time taken to train
    durA_tr=time_eA_tr-time_sA_tr
    time_sA_te=time.time()
    clf1a.predict(test_x)
    time_eA_te=time.time()
    #time taken to test
    durA_te=time_eA_te-time_sA_te
    #Accuracy on train data(model is retrained after a split of 50%/50%)
    length=len(train_x)/2
    clf1a_te=ensemble.RandomForestClassifier()
    clf1a_te.fit(train_x[:length],train_y[:length])
    acc1A=accuracy_score(train_y[length:],clf1a_te.predict(train_x[length:]))
    return acc1A,durA_tr,durA_te
    



def default_hpm():
    #Datasets
    #1)Email_spam
    path = '../../Data/Email_spam/'
    train = np.load(path + 'train.npy')
    test = np.load(path + 'test_distribute.npy')
    train_x=train[:,1:]
    train_y=train[:,0]
    test_x=test[:,1:]
    #2)Occupancy_detection
    path1 = '../../Data/Occupancy_detection/'
    train1 = np.load(path1 + 'train.npy')
    test1 = np.load(path1 + 'test_distribute.npy')
    train_x1=train1[:,1:]
    train_y1=train1[:,0]
    test_x1=test1[:,1:]
    #3)USPS_digits
    path2 = '../../Data/USPS_digits/'
    train2 = np.load(path2 + 'train.npy')
    test2 = np.load(path2 + 'test_distribute.npy')
    train_x2=train2[:,1:]
    train_y2=train2[:,0]
    test_x2=test2[:,1:]
    
    
    #1)KNN
    #a)Email_spam
    acc1a,tr1a,te1a=default_knn(test_x, train_x, train_y)
    #b)Occupancy_detection
    acc1b,tr1b,te1b=default_knn(test_x1, train_x1, train_y1)
    #c)USPS_digits
    acc1c,tr1c,te1c=default_knn(test_x2, train_x2, train_y2)
    
    #2)Decision Tree
    #a)Email_spam
    acc2a,tr2a,te2a=default_DT(test_x, train_x, train_y)
    #b)Occupancy_detection
    acc2b,tr2b,te2b=default_DT(test_x1, train_x1, train_y1)
    #c)USPS_digits
    acc2c,tr2c,te2c=default_DT(test_x2, train_x2, train_y2)
    
    #3)Random Forest
    #a)Email_spam
    acc3a,tr3a,te3a=default_RF(test_x, train_x, train_y)
    #b)Occupancy_detection
    acc3b,tr3b,te3b=default_RF(test_x1, train_x1, train_y1)
    #c)USPS_digits
    acc3c,tr3c,te3c=default_RF(test_x2, train_x2, train_y2)
    
    acc=[acc1a,acc2a,acc3a,acc1b,acc2b,acc3b,acc1c,acc2c,acc3c]
    tr=[tr1a,tr2a,tr3a,tr1b,tr2b,tr3b,tr1c,tr2c,tr3c]
    te=[te1a,te2a,te3a,te1b,te2b,te3b,te1c,te2c,te3c]
    names=['KNN','Decision_Tree','Random_Forest']
    plot_bar(acc, 'Accuracy', 'Classifiers', 'Accuracy_rate', names)
    plot_bar(tr, 'Training_Time', 'Classifiers', 'Time(s)', names)
    plot_bar(te, 'Test_Time', 'Classifiers', 'Time(s)', names)
    print "Bar plot images generated"
    

#Plot barplots
def plot_bar(arr,up_label,x_lab,y_lab,names):
    pos=list(range(3))
    width=0.25
    
    fig,ax=plt.subplots(figsize=(10,5))
    
    a=plt.bar(pos, arr[:3],width,alpha=0.5,color='#EE3224',label=names[0])
    b=plt.bar([p+width for p in pos],arr[3:6],width,alpha=0.5,color='#F78F1E',label=names[1])
    c=plt.bar([p+width*2 for p in pos],arr[6:9],width,alpha=0.5,color='#FFC222',label=names[2])
    ax.set_ylabel(y_lab)
    ax.set_xlabel(x_lab)
    ax.set_title(up_label)
    ax.set_xticks([p+1.5*width for p in pos])
    ax.set_xticklabels(names)
    plt.xlim(min(pos)-width,max(pos)+width*4)
    plt.ylim([0,max(arr)+(max(arr)/4)])
    plt.legend(['Email_spam','Occupancy_detection','USPS_digits'],loc='upper right')
    #puts height of bars on top(some bars cannot be seen owing to huge differences between maximum and minimum y_axis values)
    def labeler(rects):
        for rect in rects:
            height=str(round(rect.get_height(),4))
            ax.text(rect.get_x()+rect.get_width()/2.,1.05*rect.get_height(),'%s' % height,ha='center',va='bottom')

    labeler(a)
    labeler(b)
    labeler(c)
    plt.tight_layout()
    plt.savefig("../Figures/"+up_label+".png")
    
default_hpm()    

        
    
