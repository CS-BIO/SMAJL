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
from sklearn import svm
from sklearn.metrics import precision_recall_curve
from sklearn.tree import DecisionTreeClassifier
from RBM_model import RBM
from Predict import predict
    
def build_model(data_mF,data_dF,data_dm):
    num_row_pos = int(data_dm.sum())
    num_row_neg = int(data_dm.shape[0] * data_dm.shape[1] - data_dm.sum())
    num_col = data_mF.shape[1] + data_dF.shape[1]

    feature_matrix_pos = np.zeros(shape=(num_row_pos, num_col))  # (181032,250)
    feature_matrix_neg = np.zeros(shape=(num_row_neg, num_col))

    tmp_pos = 0
    tmp_neg = 0

    for i in range(data_dm.shape[0]):
        for j in range(data_dm.shape[1]):
            if data_dm[i][j] == 1:
                feature_matrix_pos[tmp_pos] = np.hstack((data_dF[i], data_mF[j]))
                tmp_pos += 1
            else:
                feature_matrix_neg[tmp_neg] = np.hstack((data_dF[i], data_mF[j]))
                tmp_neg += 1
    feature_matrix = np.vstack((feature_matrix_pos, feature_matrix_neg))
    rbm = RBM(feature_matrix.shape[1], feature_matrix.shape[1], max_epoch=50, learning_rate=0.00001)

    rbm.fit(feature_matrix)
    true_value_pos = rbm.forword(feature_matrix_pos)
    true_value_neg = rbm.forword(feature_matrix_neg)

    return true_value_pos,true_value_neg

def matrix_rowSum(mat):
    mat_new = np.zeros(shape=(mat.shape[0],mat.shape[1]))
    for i in range(mat.shape[0]):
        mat_new[i,i] = np.sum(mat[i,])
    return mat_new

def matrix_AS(mat_A,mat_S):
    mat_n = np.zeros(shape=(mat_S.shape[0],mat_S.shape[1]))
    
    for i in range(mat_A.shape[0]):
        for j in range(mat_S.shape[1]):
            mat_n[i,j] = np.sum(mat_A[i,])
    return mat_n

def max_min(matr):
    matr_max = np.max(matr)
    matr_min = np.min(matr)
    matr = (matr-matr_min)/(matr_max-matr_min)
    return matr

def matrix_fac(Ss,Sm,W,Y,dim_k):
    A = random.random(size=(Y.shape[0],dim_k))
    B = random.random(size=(Y.shape[1],dim_k))
    Snew = random.random(size=(Ss.shape[0],Ss.shape[1]))
    
    alpha = 10
    beta = 0.01
    gamma = 0.1
    delta = 0.1   
    for i in range(50):
        A = A*((Y.dot(B)+alpha*Snew.dot(A))/(A.dot(np.transpose(B)).dot(B)+alpha*matrix_rowSum(Snew).dot(A)+delta*A))
        B = B*(((np.transpose(Y)).dot(A)+beta*Sm.dot(B))/(B.dot(np.transpose(A)).dot(A)+beta*matrix_rowSum(Sm).dot(B)+delta*B))
        Snew = Snew*((alpha*A.dot(np.transpose(A))+2*gamma*W*Ss)/(alpha*matrix_AS(A,Ss)+2*gamma*W*Snew+2*delta*Snew))
    A = max_min(A)
    B = max_min(B)
    return A,B

def read_data(path):

    data = []
    for line in open(path, 'r'):
        ele = line.strip().split(" ")
        tmp = []
        for e in ele:
            if e != '':
                tmp.append(float(e))
        data.append(tmp)
    return data

if __name__ == '__main__':
    
    data_path = os.path.join(os.path.dirname(os.getcwd()),"data")
    data_mF = read_data(data_path+'\\miRNA794_feature84.txt')
    data_dF = read_data(data_path+'\\drug228_feature166.txt')
    data_dm = read_data(data_path+'\\Matrix_drug_miRNA.txt')
    
    data_Dsim = read_data(data_path+'\\PP-Similarity_Matrix_Drug_ATC2.txt')
    data_Msim = read_data(data_path+'\\SS-Similarity_Matrix_miRNA_BMA794.txt')
    data_W = read_data(data_path+'\\PP-Similarity_Matrix_W.txt')
    
    data_mF = np.array(data_mF)
    data_dF = np.array(data_dF)
    data_dm = np.array(data_dm)
    
    data_Dsim = np.array(data_Dsim)
    data_Msim = np.array(data_Msim)
    data_W = np.array(data_W)
    
    
    pos_position = np.zeros(shape=(int(np.sum(data_dm)),2))
    tmp_pos = 0
    tmp_arr = []
    for a in range(data_dm.shape[0]):
        for b in range(data_dm.shape[1]):
            if data_dm[a,b] == 1:
                pos_position[tmp_pos,0] = a
                pos_position[tmp_pos,1] = b
                tmp_arr.append(tmp_pos)
                tmp_pos +=1
                

    random.shuffle(tmp_arr) 
    tep_pos_set = tmp_arr
    num_tep = math.floor(len(tep_pos_set)*0.2)
    t = 5
    TPR_n_all = np.zeros(shape=(data_dm.shape[0] * data_dm.shape[1] - (int(data_dm.sum())-num_tep)+1, t))
    FPR_n_all = np.zeros(shape=(data_dm.shape[0] * data_dm.shape[1] - (int(data_dm.sum())-num_tep)+1, t))
    PRE_n_all = np.zeros(shape=(data_dm.shape[0] * data_dm.shape[1] - (int(data_dm.sum())-num_tep)+1, t))
    GM_n_all = np.zeros(shape=(data_dm.shape[0] * data_dm.shape[1] - (int(data_dm.sum())-num_tep)+1, t))
    ACC_n_all = np.zeros(shape=(data_dm.shape[0] * data_dm.shape[1] - (int(data_dm.sum())-num_tep)+1, t))
    SPE_n_all = np.zeros(shape=(data_dm.shape[0] * data_dm.shape[1] - (int(data_dm.sum())-num_tep)+1, t))
    for x in range(t):
        data_dm_new = np.zeros(shape = (data_dm.shape[0],data_dm.shape[1]))
        for n in range(data_dm.shape[0]):
            for m in range(data_dm.shape[1]):
                data_dm_new[n,m] = data_dm[n,m]      
        
        for j in range((x*num_tep),((x+1)*num_tep)):
            data_dm_new[int(pos_position[tep_pos_set[j],0]),int(pos_position[tep_pos_set[j],1])] = 0 

        A,B = matrix_fac(data_Dsim,data_Msim,data_W,data_dm_new,dim_k=120)  

        feature_pos1,feature_neg1 = build_model(B,A,data_dm_new) 
        feature_pos2,feature_neg2 = build_model(data_mF,data_dF,data_dm_new)  
        
        
        feature_pos12 = max_min(np.hstack((0.99*max_min(feature_pos1),0.01*max_min(feature_pos2))))  
        feature_neg12 = max_min(np.hstack((0.99*max_min(feature_neg1),0.01*max_min(feature_neg2))))
        
    
        feature_matrix = np.vstack((feature_pos12,feature_neg12))  
        rbm = RBM(feature_matrix.shape[1], feature_matrix.shape[1], max_epoch=50, learning_rate=0.00001)  

        rbm.fit(feature_matrix)
        feature_pos = rbm.forword(feature_pos12)
        feature_neg = rbm.forword(feature_neg12)
   
        tpr_n, fpr_n, pre_n = predict.predict_model(feature_pos,feature_neg,data_dm,data_dm_new)

