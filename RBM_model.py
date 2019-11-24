# !/usr/bin/python
# -*- coding: UTF-8 -*-
# -*- coding: utf-8 -*-

import matplotlib.pylab as plt
import numpy as np
import random,math
import pylab as pl
import os
from random import randint, sample
from numpy import trapz
from sklearn import svm
from sklearn.metrics import precision_recall_curve


class RBM(object):

    def __init__(self, n_visible, n_hidden, momentum=0.5, learning_rate=0.1, max_epoch=50, batch_size=128, penalty=0,
                 weight=None, v_bias=None, h_bias=None):

        self.n_visible = n_visible
        self.n_hidden = n_hidden
        self.max_epoch = max_epoch
        self.batch_size = batch_size
        self.penalty = penalty
        self.learning_rate = learning_rate
        self.momentum = momentum

        if weight is None:
            self.weight = np.random.random((self.n_hidden, self.n_visible)) * 0.1  
        else:
            self.weight = weight
        if v_bias is None:
            self.v_bias = np.zeros(self.n_visible)  
        else:
            self.v_bias = v_bias
        if h_bias is None:
            self.h_bias = np.zeros(self.n_hidden)  
        else:
            self.h_bias = h_bias

    def sigmoid(self, z):
        return 1.0 / (1.0 + np.exp(-z))

    def forword(self, inpt):
        z = np.dot(inpt, self.weight.T) + self.h_bias  
        return self.sigmoid(z)

    def backward(self, inpt):
        z = np.dot(inpt, self.weight) + self.v_bias  
        return self.sigmoid(z)

    def batch(self):
        m, n = self.input_x.shape
        per = list(range(m))
        random.shuffle(per)
        per = [per[k:k + self.batch_size] for k in range(0, m, self.batch_size)]
        batch_data = []
        for group in per:
            batch_data.append(self.input_x[group])
        return batch_data

    def fit(self, input_x):
        self.input_x = input_x
        Winc = np.zeros_like(self.weight)
        binc = np.zeros_like(self.v_bias)
        cinc = np.zeros_like(self.h_bias)

        for epoch in range(self.max_epoch):
            batch_data = self.batch()
            num_batchs = len(batch_data)
            err_sum = 0.0
            self.penalty = (1 - 0.9 * epoch / self.max_epoch) * self.penalty
            for v0 in batch_data:
                h0 = self.forword(v0)
                h0_states = np.zeros_like(h0)
                h0_states[h0 > np.random.random(h0.shape)] = 1

                v1 = self.backward(h0_states)
                v1_states = np.zeros_like(v1)
                v1_states[v1 > np.random.random(v1.shape)] = 1

                h1 = self.forword(v1_states)
                h1_states = np.zeros_like(h1)
                h1_states[h1 > np.random.random(h1.shape)] = 1

                dW = np.dot(h0_states.T, v0) - np.dot(h1_states.T, v1)
                db = np.sum(v0 - v1, axis=0).T
                dc = np.sum(h0 - h1, axis=0).T

                Winc = self.momentum * Winc + self.learning_rate * (dW - self.penalty * self.weight) / self.batch_size
                binc = self.momentum * binc + self.learning_rate * db / self.batch_size
                cinc = self.momentum * cinc + self.learning_rate * dc / self.batch_size

                self.weight = self.weight + Winc
                self.v_bias = self.v_bias + binc
                self.h_bias = self.h_bias + cinc

                err_sum = err_sum + np.mean(np.sum((v0 - v1)**2, axis=1))

            err_sum = err_sum / num_batchs
            if (epoch % 10 == 0):
              print('Epoch {0},err_sum {1}'.format(epoch, err_sum))
    
    def predict(self,input_x):   

        h0 = self.forword(input_x)                
        h0_states = np.zeros_like(h0)                        
        h0_states[h0 > np.random.random(h0.shape)] = 1                        
        
        v1 = self.backward(h0_states)
        return v1
   