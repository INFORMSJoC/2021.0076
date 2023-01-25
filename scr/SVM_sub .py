# -*- coding: utf-8 -*-


import numpy as np
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import IsolationForest
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.semi_supervised import SelfTrainingClassifier
from scipy.optimize import minimize
from copy import deepcopy
from co_forest.coforest import coforest
from xgboost import XGBClassifier
from MLP_VAE.MLP_VAE import MLP_VAE
from semi_GAN_mine import creat_data
import time


num_test = 100 #Only a small number of adversarial samples are generated for quick debugging
num_change = 10 #number of modifiable features


def distEclud(vecA, vecB):
    return np.sqrt(sum(np.power(vecA - vecB, 2))) #la.norm(vecA-vecB)

def kenel(gamma,sv_i,x):
    dist = distEclud(sv_i,x)
    return np.e**(-gamma*dist)
    
def normalTarget(args, num_change):
    '''
    Args:
        current_sample: Fraud samples that need to be modified
        dual_coef: dual coefficients of the substitute SVM 
        sv: support vectors
        intercept: intercept
    
    Return:
        target function    
    '''
    current_sample, dual_coef, sv, intercept = args
    def fun(x):
        zero_features = np.zeros(len(current_sample)-num_change)
        x_new = np.hstack((x,zero_features))
        total = 0
        gamma = len(current_sample)**(-1)
        for i in range(len(sv)):
            total += dual_coef[i]*kenel(gamma,sv[i],current_sample + x_new)
        return total+intercept
    return fun
    
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

        
    def GANfit(self,trainData,trainLabel, unlabelRate):
        trainData = deepcopy(trainData)
        trainLabel = deepcopy(trainLabel)

        X_total, y_total = creat_data(trainData, trainLabel, unlabelRate)
        
        self.clf.fit(X_total, y_total)
        
    def fit(self,trainData,trainLabel, unlabelRate):
        sss = StratifiedShuffleSplit(n_splits=1,test_size=unlabelRate, random_state=1)  
        
        for train_index, test_index in sss.split(trainData,trainLabel):
            X_tr = trainData[train_index]
            y_tr = trainLabel[train_index]      
        self.clf.fit(X_tr,y_tr)
  
    def selffit(self,trainData,trainLabel, unlabelRate):
        trainData = deepcopy(trainData)
        trainLabel = deepcopy(trainLabel)
        sss = StratifiedShuffleSplit(n_splits=1,test_size=unlabelRate, random_state=1)         
        for train_index, test_index in sss.split(trainData,trainLabel):
            trainLabel[test_index] = -1
        
        self_training_model = SelfTrainingClassifier(SVC(kernel='rbf',probability=1,C=1))

        semi_clf = self_training_model.fit(trainData,trainLabel)
        self.clf = semi_clf.base_estimator_

    def CFfit(self,trainData,trainLabel, unlabelRate):
        trainData = deepcopy(trainData)
        trainLabel = deepcopy(trainLabel)
        sss = StratifiedShuffleSplit(n_splits=1,test_size=unlabelRate, random_state=1)         
        for train_index, test_index in sss.split(trainData,trainLabel):
            trainLabel[test_index] = -1

            
        self.penalSamples = trainData[np.where(trainLabel==0)]

        selfclf = coforest(clf=DecisionTreeClassifier(), finalClf=SVC(kernel='rbf',random_state=1,probability=1,C=1))
        selfclf.fit(trainData,trainLabel)
        self.clf = selfclf.finalClf        

    def optimizeCoff(self,X,y,target=normalTarget):
        """
        Description:
        Modify the adversarial samples in X to make them normal samples

        Parameters:
        X - array-like: input dataset
        y - array-like: The category label of the samples, 0 : normal sample, 1 : fraudulent sample
        target: target function
        """        
        X = deepcopy(X)
        fault_samples_index = np.where(y==1)[0][:num_test]
        cons1 = ({'type': 'ineq','fun' : lambda x: self.d-np.sum((x)**2)**0.5})
        
        fault_samples = X[fault_samples_index]
        
        init_SuccessesLR = len(fault_samples)-np.sum(self.oracleLR.predict(fault_samples))
        init_SuccessesDT = len(fault_samples)-np.sum(self.oracleDT.predict(fault_samples))
        init_SuccessesRF = len(fault_samples)-np.sum(self.oracleRF.predict(fault_samples))
        init_SuccessesSVM = len(fault_samples)-np.sum(self.oracleSVM.predict(fault_samples))
        init_SuccessesMLP = len(fault_samples)-np.sum(self.oracleMLP.predict(fault_samples))
        init_SuccessesIF = len(fault_samples)-np.sum(self.oracleIF.predict(fault_samples)==-1) #该算法输出正类为-1
        init_SuccessesVAE = len(fault_samples)-np.sum(self.oracleVAE.predict(fault_samples))

        self.init_accLR  = 1-float(init_SuccessesLR)/len(fault_samples)
        self.init_accDT  = 1-float(init_SuccessesDT)/len(fault_samples)
        self.init_accRF  = 1-float(init_SuccessesRF)/len(fault_samples)
        self.init_accSVM  = 1-float(init_SuccessesSVM)/len(fault_samples)
        self.init_accMLP  = 1-float(init_SuccessesMLP)/len(fault_samples)
        self.init_accIF  = 1-float(init_SuccessesIF)/len(fault_samples)
        self.init_accVAE  = 1-float(init_SuccessesVAE)/len(fault_samples)          
        
             
        for index in fault_samples_index:
            
            
            print(index)
            x = np.zeros(num_change)
            #args = (X[index], self.clf.coef_, self.penalSamples) 
            
            dual_coef = self.clf.dual_coef_[0]
            sv = self.clf.support_vectors_
            intercept = self.clf.intercept_
            
            args = (X[index], dual_coef, sv, intercept)

            
            res = minimize(target(args, num_change), x, method='SLSQP',
                           constraints=cons1,options={'maxiter': 5,'disp': True},tol = 1e-5)
            X[index, :num_change] = X[index, :num_change] + res.x  


        fault_samples = X[fault_samples_index]
        np.clip(fault_samples, 0.0, 1.0)
        
        numOfSuccessesLR = len(fault_samples)-np.sum(self.oracleLR.predict(fault_samples))
        numOfSuccessesDT = len(fault_samples)-np.sum(self.oracleDT.predict(fault_samples))
        numOfSuccessesRF = len(fault_samples)-np.sum(self.oracleRF.predict(fault_samples))
        numOfSuccessesSVM = len(fault_samples)-np.sum(self.oracleSVM.predict(fault_samples))
        numOfSuccessesMLP = len(fault_samples)-np.sum(self.oracleMLP.predict(fault_samples))
        numOfSuccessesIF = len(fault_samples)-np.sum(self.oracleIF.predict(fault_samples)==-1) #it outputs a positive class as -1
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


    print('len(X_te)',len(X_te))
    fault_samples_index = np.where(y_te==1)[0]
    fault_samples = X_te[fault_samples_index][:num_test]
    print('len(fault_samples):',len(fault_samples))      
    oracleLR = LogisticRegression(random_state=0,fit_intercept=0).fit(X_tr, y_tr)
    oracleDT = DecisionTreeClassifier(random_state=0).fit(X_tr, y_tr)
    oracleRF = XGBClassifier(max_depth=3, learning_rate=0.2, n_estimators=100, 
                             objective='binary:logistic',random_state=0).fit(X_tr,y_tr)      
    oracleSVM = SVC(random_state=0,probability=1,C=1).fit(X_tr, y_tr)
    oracleMLP = MLPClassifier(solver='adam', alpha=1e-5,max_iter=3000, hidden_layer_sizes=(10,10), random_state=0).fit(X_tr, y_tr)
    oracleIF = IsolationForest(max_samples = 'auto', contamination=0.3).fit(X_tr)
    oracleVAE = MLP_VAE(X_tr.shape[1], 20, 0.3)
    oracleVAE.train(X_tr)

    
       
    AUC_listLR = []
    AUC_listDT = []
    AUC_listRF = []
    AUC_listSVM = []
    AUC_listMLP = []
    AUC_listIF = []
    AUC_listVAE = []
    
    SVM = substituteLR(subclf=SVC(kernel='rbf',probability=1,C=1),
                      oracleLR=oracleLR,
                      oracleDT=oracleDT,
                      oracleRF=oracleRF,
                      oracleSVM=oracleSVM,
                      oracleMLP=oracleMLP,
                      oracleIF=oracleIF,
                      oracleVAE=oracleVAE)
    time1 = time.time()
    exec('SVM.{name}(X_tr,y_tr,0.95)'.format(name=method))

    for i in np.arange(0.1,2.1,0.3):
        

        SVM.d = i
        init_acc = []

        SVM.optimizeCoff(X_te,y_te,target = normalTarget)
        
        time2 = time.time()
        print('time:', time2-time1)
        init_acc.append([SVM.init_accLR, SVM.init_accDT, SVM.init_accRF, SVM.init_accSVM, SVM.init_accMLP, SVM.init_accIF, SVM.init_accVAE])

        AUC_listLR.append(SVM.accLR)
        AUC_listDT.append(SVM.accDT)
        AUC_listRF.append(SVM.accRF)
        AUC_listSVM.append(SVM.accSVM)
        AUC_listMLP.append(SVM.accMLP)
        AUC_listIF.append(SVM.accIF)
        AUC_listVAE.append(SVM.accVAE)

    return init_acc, AUC_listLR, AUC_listDT, AUC_listRF,  AUC_listSVM, AUC_listMLP, AUC_listIF, AUC_listVAE











 