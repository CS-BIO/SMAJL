# !/usr/bin/python
# -*- coding: UTF-8 -*-
# -*- coding: utf-8 -*-

import matplotlib.pylab as plt
import numpy as np
import random,math
import pylab as pl
import os
from random import randint, sample
from numpy import trapz,random
from numpy import *
from sklearn import svm,tree,ensemble
from sklearn.naive_bayes import MultinomialNB,BernoulliNB,GaussianNB
from sklearn.metrics import precision_recall_curve,roc_curve,auc
from sklearn.linear_model import LogisticRegression
from RBM_model import RBM
from sklearn.neighbors import KNeighborsClassifier

class predict():
    def predict_model(feature_pos,feature_neg,data_dm,data_dm_new):

        pos_AllFeature_train = np.zeros(shape=(feature_pos.shape[0], feature_pos.shape[1]))
        neg_AllFeature_train = np.zeros(shape=(feature_pos.shape[0], feature_pos.shape[1]))
        
        date_neg = [randint(0, feature_neg.shape[0]-1) for _ in range(feature_neg.shape[0])]
        tep_neg_set = sample(date_neg, pos_AllFeature_train.shape[0])
        for aa in range(len(tep_neg_set)):
            neg_AllFeature_train[aa] = feature_neg[tep_neg_set[aa]]
            
        for bb in range(feature_pos.shape[0]):
            pos_AllFeature_train[bb] = feature_pos[bb]

        X_train = np.vstack((pos_AllFeature_train,neg_AllFeature_train))  
        y_train = np.hstack(([1]*pos_AllFeature_train.shape[0],[0]*neg_AllFeature_train.shape[0]))

        X_test = feature_neg
        y_test = [0]*feature_neg.shape[0]
        
        tmp_pos = 0
        for a in range(data_dm_new.shape[0]):
            for b in range(data_dm_new.shape[1]):
                if data_dm[a,b] == 1 and data_dm_new[a,b]== 0:
                    y_test[tmp_pos] = 1
                    tmp_pos +=1
                elif data_dm_new[a,b] == 0:
                    tmp_pos += 1

        model = LogisticRegression()
        model.fit(X_train,y_train)
        y_score = model.predict_proba(X_test)[:,1]

        data_ROC_n = np.vstack((y_score, y_test)).T
        data_ROC_n = data_ROC_n[np.lexsort(-data_ROC_n[:,::-1].T)]

        n = len(data_ROC_n)
        tpr_n = []
        fpr_n = []
        pre_n = []
        gm_n = []
        spe_n = []
        tp_fn = sum(data_ROC_n[:,1])
        tn_fp = len(data_ROC_n)-tp_fn

        for j in range(1,n+1):
           if (j % 50000 == 0):
              print(j)
           tp_n = sum(data_ROC_n[1:j,1])
           fp_n = j - tp_n
           tn_n = tn_fp - fp_n  
           fn_n = tp_fn - tp_n  
           tpr_n.append(tp_n/tp_fn)
           fpr_n.append(fp_n/tn_fp)
           pre_n.append(tp_n/(tp_n + fp_n))

           gm_n.append(((tp_n / (tp_n + fn_n)) * (tn_n / (fp_n + tn_n))) ** 0.5)
           spe_n.append(tn_n / (tn_n + fp_n))

        tpr_n.append(1)
        fpr_n.append(1)
        pre_n.append(0)
        gm_n.append(1)
        spe_n.append(1)

        return tpr_n,fpr_n,pre_n



