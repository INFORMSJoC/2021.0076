# -*- coding: utf-8 -*-

import numpy as np
import random
import copy
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression

class coforest(object):
    def __init__(self, clf, finalClf, n_tree=6, p=-1, n=-1, k=20, theta = 0.75):
        self.finalClf = finalClf
        self.clf = clf
        if (p == -1 and n != -1) or (p != -1 and n == -1):
            raise ValueError('Current implementation supports either both p and n being specified, or neither')  
        self.p_ = p
        self.n_ = n
        self.theta = theta
        self.n_tree = n_tree
        random.seed()
        
    def fit(self, X, y):
        """
        Description:
            fits the classifiers on the partially labeled data, y.

        Parameters:
            X - array-like (n_samples, n_features): training samples
            y - array-like (n_samples): labels for samples, -1 indicates unlabeled

        """

        #we need y to be a numpy array so we can do more complex slicing
        y = np.asarray(y)
        self.y = y
        #set the n and p parameters if we need to
        if self.p_ == -1 and self.n_ == -1:
            num_pos = sum(1 for y_i in y if y_i == 1)
            num_neg = sum(1 for y_i in y if y_i == 0)
			
            n_p_ratio = num_neg / float(num_pos)
		
            if n_p_ratio > 1:
                self.p_ = 1
                self.n_ = round(self.p_*n_p_ratio)

            else:
                self.n_ = 1
                self.p_ = round(self.n_/n_p_ratio)

        assert(self.p_ > 0 and self.n_ > 0)
        
        #the set of unlabeled samples
        U = np.array([i for i, y_i in enumerate(y) if y_i == -1])
        
        #we randomize here, and then just take from the back so we don't have to sample every time
        random.shuffle(U)
        
        #the samples that are initially labeled
        L = np.array([i for i, y_i in enumerate(y) if y_i != -1])
        self.L = L
        
        for i in range(self.n_tree):
            exec('self.tree{name}=copy.copy(self.clf)'.format(name=i)) #指定分类模型
            exec('self.y{name}=copy.copy(self.y)'.format(name=i)) #指定分类模型
            
            
            index = np.random.randint(len(L),size=int(len(L)*0.7)) #随机选取70%的初始有标签数据
            L_train = L[index]
            exec('self.L{name}=copy.deepcopy(L_train)'.format(name=i)) #指定每个基分类自己的L
            exec('self.tree{name}.fit(X[L_train],y[L_train])'.format(name=i))
        
        
        
        
        it = 0
        num_U = 10 #池子里面的样本数量
        while it != 5:
#            print('it',it)
            it += 1
            for i in range(self.n_tree): 
                index = np.random.randint(len(U),size=num_U)   
                exec('self.U_{name}=copy.deepcopy(U)[index]'.format(name=i)) #为每个基分类器选择未标记样本池
            for i in range(self.n_tree):
                y_pred_prob = np.zeros((num_U,2))
                
                for j in list(range(self.n_tree)[:i]) + list(range(self.n_tree)[i+1:]):     #对于其余基分类器
#                    print(self.tree1.predict_proba(X[self.U_0]))
                    y_pred_prob = y_pred_prob + eval('y_pred_prob + self.tree{name1}.predict_proba(X[self.U_{name2}])'.format(name1=j,name2=i))
          
#                    print(y_pred_prob)
                y_pred_prob = y_pred_prob/(self.n_tree-1) #计算其余基分类器对U_i的置信度累计之和
                best_n = np.where(y_pred_prob[:,0]>=self.theta) #最好的负类样本在U_i中的索引
                best_p = np.where(y_pred_prob[:,1]>=self.theta) #最好的正类样本在U_i中的索引
                
                exec('self.y{name1}[self.U_{name2}[best_n]] = 0'.format(name1=i,name2=i)) #为选出的负负类样本赋予标签
                exec('self.y{name1}[self.U_{name2}[best_p]] = 1'.format(name1=i,name2=i)) #为选出的正负类样本赋予标签                

                #exec('print self.L{name}.shape'.format(name=i))
                exec('self.L{name} = np.hstack((self.L{name},self.U_{name}[best_n]))'.format(name=i)) #扩展负类样本到L
                #exec('print self.L{name}.shape'.format(name=i))
                exec('self.L{name} = np.hstack((self.L{name},self.U_{name}[best_p]))'.format(name=i)) #扩展正类样本到L
                #exec('print self.L{name}.shape'.format(name=i))
                #print self.y0[self.L0]                
                exec('self.tree{name}.fit(X[self.L{name}],self.y{name}[self.L{name}])'.format(name=i)) #重新训练基分类器
        
        L_ = []      #L_表示所有基分类器的最后的有类别标记样本的索引
        for i in range(self.n_tree):
            exec('L_.extend(list(self.L{name}))'.format(name=i))
        L_set = set(L_)
        newU = L_set-set(self.L)   #newU是所有基分类器的新增样本的索引
        newU = np.array(list(newU))
        self.newU = newU
        
        oriX = X[self.L]  #初始有标签样本集
        oriy = y[self.L] #初始有标签样本的类别标签
        newX = X[self.newU] #新增有标签样本集
        newy = self.predict(X[self.newU]) #新增有标签样本的类别标签（基分类投票）
        
        finalX = np.vstack((oriX,newX))   #最终用来LR的训练集
        finaly = np.hstack((oriy,newy))  
        
        self.finalClf.fit(finalX,finaly)
        
        #print finalX.shape, finaly.shape
        
        
    def predict(self,X):
        result = []
        for i in range(self.n_tree):
            exec('result.append(self.tree{name}.predict(X))'.format(name=i))
        result = np.array(result)
        #print result
        num_p = np.sum(result==1,0)
        
        num_n = np.sum(result==0,0)
        vote = np.zeros(len(num_p))
        #print vote
        for i in range(len(num_p)):
            if num_p[i]>num_n[i]:
                vote[i] = 1
            else:
                vote[i] = 0   
        return vote

