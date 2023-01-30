# -*- coding: utf-8 -*-

import numpy as np
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC,SVC
from sklearn.ensemble import  IsolationForest
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.semi_supervised import SelfTrainingClassifier
from scipy.optimize import minimize
from copy import deepcopy
from co_forest.coforest import coforest
from xgboost import XGBClassifier
from MLP_VAE.MLP_VAE import MLP_VAE
from GAN.semi_GAN import creat_data
 

num_test = 100 #Only a small number of adversarial samples are generated for quick debugging
num_change = 10 #number of modifiable features

class substituteLR(object):
    def __init__(self, subclf, oracleLR, oracleDT, oracleRF, oracleSVM, oracleMLP, oracleIF, oracleVAE, d=0.5):
        
        self.clf = subclf
        self.oracleLR = oracleLR
        self.oracleDT = oracleDT
        self.oracleRF = oracleRF
        self.oracleSVM = oracleSVM
        self.oracleMLP = oracleMLP
        self.oracleIF = oracleIF
        self.oracleVAE = oracleVAE
        self.d = d
             
    def fit(self,trainData,trainLabel, unlabelRate):
        """
        without using semi-supervised learning
        """         
        
        sss = StratifiedShuffleSplit(n_splits=1,test_size=unlabelRate, random_state=1)     
        for train_index, test_index in sss.split(trainData,trainLabel):
            X_tr = trainData[train_index]
            y_tr = trainLabel[train_index]      
           
        self.penalSamples = trainData[np.where(trainLabel==0)]      
        X_tr = X_tr[:,:num_change]
        self.clf.fit(X_tr,y_tr)

     
    def GANfit(self,trainData,trainLabel, unlabelRate):
        """
        semi-supervised GAN
        """ 
        trainData = deepcopy(trainData)
        trainLabel = deepcopy(trainLabel)

        X_total, y_total = creat_data(trainData, trainLabel, unlabelRate)
        
        self.clf.fit(X_total, y_total)
        
    def selffit(self,trainData,trainLabel, unlabelRate):
        """
        self-labeling
        """  
        trainData = deepcopy(trainData)
        trainLabel = deepcopy(trainLabel)
        
        sss = StratifiedShuffleSplit(n_splits=1,test_size=unlabelRate, random_state=1)         
        for train_index, test_index in sss.split(trainData,trainLabel):
            trainData = trainData[test_index]   
            trainLabel = trainLabel[test_index]
        
        sss = StratifiedShuffleSplit(n_splits=1,test_size=unlabelRate, random_state=1)         
        for train_index, test_index in sss.split(trainData,trainLabel):
            trainLabel[test_index] = -1

        self_training_model = SelfTrainingClassifier(SVC(kernel='linear',probability=1,C=1))


        trainData = trainData[:,:num_change]
        semi_clf = self_training_model.fit(trainData,trainLabel)
        self.clf = semi_clf.base_estimator_

    def CFfit(self,trainData,trainLabel, unlabelRate):
        """
        Co-forest
        """  
        trainData = deepcopy(trainData)
        trainLabel = deepcopy(trainLabel)
        sss = StratifiedShuffleSplit(n_splits=1,test_size=unlabelRate, random_state=1)         
        for train_index, test_index in sss.split(trainData,trainLabel):
            trainLabel[test_index] = -1
        trainData = trainData[:,:num_change]
        selfclf = coforest(clf=DecisionTreeClassifier(), finalClf=LogisticRegression(max_iter=5000))
        selfclf.fit(trainData,trainLabel)
        self.clf = selfclf.finalClf        

    def optimizeCoff(self,X,y):
        """
        Description:
        Modify the positive samples in X to make them normal samples

        Parameters:
        X - array-like: input dataset
        y - array-like: The category label of the samples, 0 : normal sample, 1 : fraudulent sample
        """      
        X = deepcopy(X)
        fault_samples_index = np.where(y==1)[0][:num_test]       
        fault_samples = X[fault_samples_index]
        
        init_SuccessesLR = len(fault_samples)-np.sum(self.oracleLR.predict(fault_samples))
        init_SuccessesDT = len(fault_samples)-np.sum(self.oracleDT.predict(fault_samples))
        init_SuccessesRF = len(fault_samples)-np.sum(self.oracleRF.predict(fault_samples))
        init_SuccessesSVM = len(fault_samples)-np.sum(self.oracleSVM.predict(fault_samples))
        init_SuccessesMLP = len(fault_samples)-np.sum(self.oracleMLP.predict(fault_samples))
        init_SuccessesIF = len(fault_samples)-np.sum(self.oracleIF.predict(fault_samples)==-1) #it outputs a positive class as -1
        init_SuccessesVAE = len(fault_samples)-np.sum(self.oracleVAE.predict(fault_samples))

        self.init_accLR  = 1-float(init_SuccessesLR)/len(fault_samples)
        self.init_accDT  = 1-float(init_SuccessesDT)/len(fault_samples)
        self.init_accRF  = 1-float(init_SuccessesRF)/len(fault_samples)
        self.init_accSVM  = 1-float(init_SuccessesSVM)/len(fault_samples)
        self.init_accMLP  = 1-float(init_SuccessesMLP)/len(fault_samples)
        self.init_accIF  = 1-float(init_SuccessesIF)/len(fault_samples)
        self.init_accVAE  = 1-float(init_SuccessesVAE)/len(fault_samples)    
        
        
        w = self.clf.coef_
        perturbation = (-1*self.d*w)/np.linalg.norm(w, ord=2)
        perturbation = perturbation[0]
          
        for index in fault_samples_index:          
            X[index, :num_change] = X[index, :num_change] + perturbation[:num_change]


        fault_samples = X[fault_samples_index]
        np.clip(fault_samples, 0.0, 1.0)
        
        numOfSuccessesLR = len(fault_samples)-np.sum(self.oracleLR.predict(fault_samples))
        numOfSuccessesDT = len(fault_samples)-np.sum(self.oracleDT.predict(fault_samples))
        numOfSuccessesRF = len(fault_samples)-np.sum(self.oracleRF.predict(fault_samples))
        numOfSuccessesSVM = len(fault_samples)-np.sum(self.oracleSVM.predict(fault_samples))
        numOfSuccessesMLP = len(fault_samples)-np.sum(self.oracleMLP.predict(fault_samples))
        numOfSuccessesIF = len(fault_samples)-np.sum(self.oracleIF.predict(fault_samples)==-1) #The algorithm outputs a positive class as -1
        numOfSuccessesVAE = len(fault_samples)-np.sum(self.oracleVAE.predict(fault_samples))

        self.accLR  = 1-float(numOfSuccessesLR)/len(fault_samples)
        self.accDT  = 1-float(numOfSuccessesDT)/len(fault_samples)
        self.accRF  = 1-float(numOfSuccessesRF)/len(fault_samples)
        self.accSVM  = 1-float(numOfSuccessesSVM)/len(fault_samples)
        self.accMLP  = 1-float(numOfSuccessesMLP)/len(fault_samples)
        self.accIF  = 1-float(numOfSuccessesIF)/len(fault_samples)
        self.accVAE  = 1-float(numOfSuccessesVAE)/len(fault_samples)        
        
def test(X,y,method):    
    sss = StratifiedShuffleSplit(n_splits=1,test_size=0.3, random_state=0)              
    for train_index, test_index in sss.split(X,y):
        X_tr = X[train_index]
        y_tr = y[train_index]
        X_te = X[test_index]
        y_te = y[test_index]

    X_n = X_tr[np.where(y_tr==0)]
    print('len(X_te)',len(X_te))
    fault_samples_index = np.where(y_te==1)[0]
    fault_samples = X_te[fault_samples_index][:num_test]
    print('len(fault_samples):',len(fault_samples))    

    
    oracleLR = LogisticRegression(random_state=0,fit_intercept=0).fit(X_tr, y_tr)
    oracleDT = DecisionTreeClassifier(random_state=0).fit(X_tr, y_tr)
    oracleRF = XGBClassifier(max_depth=3, learning_rate=0.2, n_estimators=100, 
                             objective='binary:logistic',random_state=0).fit(X_tr,y_tr)  
    oracleSVM = SVC(C=1,random_state=0).fit(X_tr, y_tr)
    oracleMLP = MLPClassifier(solver='adam', alpha=1e-5,max_iter=3000, hidden_layer_sizes=(10,10), random_state=0).fit(X_tr, y_tr)
    oracleIF = IsolationForest(n_estimators=100, max_samples = 500,max_features=10, contamination=0.3).fit(X_tr)
    
    oracleVAE = MLP_VAE(X_tr.shape[1], 20, 0.3)    
    oracleVAE.train(X_n)    
    
    acc_LR = []
    acc_DT = []
    acc_RF = []
    acc_SVM = []
    acc_MLP = []
    acc_IF = []
    acc_VAE = []
    
    LR = substituteLR(subclf=LinearSVC(C=1),
                      oracleLR=oracleLR,
                      oracleDT=oracleDT,
                      oracleRF=oracleRF,       
                      oracleSVM=oracleSVM,
                      oracleMLP=oracleMLP,
                      oracleIF=oracleIF,
                      oracleVAE=oracleVAE)

    exec('LR.{name}(X_tr,y_tr,0.95)'.format(name=method))

    for i in np.arange(0.1,2.1,0.1):
        print(i)       
        LR.d = i
        init_acc = []
        LR.optimizeCoff(X_te,y_te)
        init_acc.append([LR.init_accLR, LR.init_accDT, LR.init_accRF, LR.init_accSVM, 
                         LR.init_accMLP, LR.init_accIF, LR.init_accVAE])
        acc_LR.append(LR.accLR)
        acc_DT.append(LR.accDT)
        acc_RF.append(LR.accRF)
        acc_SVM.append(LR.accSVM)
        acc_MLP.append(LR.accMLP)
        acc_IF.append(LR.accIF)
        acc_VAE.append(LR.accVAE)

    return init_acc, acc_LR, acc_DT, acc_RF, acc_SVM, acc_MLP, acc_IF, acc_VAE







 