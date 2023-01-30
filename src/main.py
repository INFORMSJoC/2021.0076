# -*- coding: utf-8 -*-
"""
Created on Thu Apr 11 19:59:01 2019

@author: coco
"""

import numpy as np
from sklearn.preprocessing import MinMaxScaler
from imblearn.under_sampling import RandomUnderSampler
from linear_sub_part import test
   
if __name__ == '__main__': 
    Data = np.loadtxt('toy_data.txt') #没有重抽样的数据集
    
    X, y = Data[:,:-1], Data[:,-1]
    scaler = MinMaxScaler()
    scaler.fit(X)
    X = scaler.transform(X)
    RUS = RandomUnderSampler(sampling_strategy = 0.3)
    X, y = RUS.fit_sample(X,y)  
    
    total_listLR = []
    total_listDT = []
    total_listRF = []
    total_listSVM = []
    total_listMLP = []
    total_listIF = []
    total_listVAE = []
    
    for i in range(10):
        print(i)
    
        init_acc, m1, m2, m3, m5, m6, m7, m8 = test(X,y,'GANfit')       
        total_listLR.append(m1)
        total_listDT.append(m2)
        total_listRF.append(m3)
        total_listSVM.append(m5)
        total_listMLP.append(m6)
        total_listIF.append(m7)
        total_listVAE.append(m8)
        print('----------------------------')
    
    total_listLR = np.array(total_listLR).T
    total_listDT = np.array(total_listDT).T
    total_listRF = np.array(total_listRF).T
    total_listSVM = np.array(total_listSVM).T
    total_listMLP = np.array(total_listMLP).T
    total_listIF = np.array(total_listIF).T
    total_listVAE = np.array(total_listVAE).T
                            
    min1 = np.average(total_listLR,1)
    min2 = np.average(total_listDT,1)
    min3 = np.average(total_listRF,1)
    min5 = np.average(total_listSVM,1)
    min6 = np.average(total_listMLP,1)
    min7 = np.average(total_listIF,1)      
    min8 = np.average(total_listVAE,1)    
                      
    result = np.vstack((min1,min2,min3,min5,min6,min7,min8))
    result = result.T 
    np.savetxt('LR_sub_acc.txt',result)
    print(np.around(result,4))   







 