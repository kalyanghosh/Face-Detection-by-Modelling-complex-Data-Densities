# -*- coding: utf-8 -*-
"""

@author: Kslyan

"""
import cv2
import glob
from sklearn.decomposition import PCA
from scipy.stats import multivariate_normal
import numpy as np
from scipy import special
from math import *
import os
import xml.etree.ElementTree as ET
import cv2
import glob
import numpy as np
import csv
import math
from matplotlib.mlab import PCA
from sklearn import decomposition
from sklearn.decomposition import PCA 
from sklearn.preprocessing import StandardScaler
from scipy import linalg
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
dof=4.0
count=0
testsize=100
trainsize=1000
n_components=60
from sklearn.metrics import roc_curve, auc
test_size=100


def readImage(filename):
    image_list=[]
    for file in glob.glob(filename):
        im=cv2.imread(file)
        img = cv2.cvtColor(im,cv2.COLOR_RGB2GRAY )
        img=cv2.resize(img,(60,60))
        img=img.flatten()
        image_list.append(img)
    image_array=np.asarray(image_list)
    return image_array

trainingFaces=readImage('C:\\Users\\kghosh\\Downloads\\Latest_CV_Codes_28th\\Latest_CV_Codes_28th\\CV_Project_763\\Train_FaceData\\*.jpg')
trainingNonFaces=readImage('C:\\Users\\kghosh\\Downloads\\Latest_CV_Codes_28th\\Latest_CV_Codes_28th\\CV_Project_763\\Train_NonFaceData\\*.jpg')
testingface=readImage('C:\\Users\\kghosh\\Downloads\\Latest_CV_Codes_28th\\Latest_CV_Codes_28th\\CV_Project_763\\Test_FaceData\\*.jpg')
testingnonface=readImage('C:\\Users\\kghosh\\Downloads\\Latest_CV_Codes_28th\\Latest_CV_Codes_28th\\CV_Project_763\\Test_NonFaceData\\*.jpg')

def fitting_T(x,mu,Sigma,df,d):
   
    Num = gamma(1. * (d+df)/2)
    Denom = ( gamma(1.*df/2) * pow(df*pi,1.*d/2) * pow(np.linalg.det(Sigma),1./2) * pow(1 + (1./df)*np.dot(np.dot((x - mu),np.linalg.inv(Sigma)), (x - mu)),1.* (d+df)/2))
    d = 1. * Num / Denom 
    return d


def T_distribution(X, dof=3.5, iter=100, eps=1e-1):
    
    # initialize parameters
    D = X.shape[1]
    N = X.shape[0]
    cov = np.cov(X,rowvar=False)
    mean = X.mean(axis=0)
    mu = X - mean[None,:]
    delta = np.einsum('ij,ij->i', mu, np.linalg.solve(cov,mu.T).T)
    z = (dof + D) / (dof + delta)
    obj = [
        -N*np.linalg.slogdet(cov)[1]/2 - (z*delta).sum()/2 \
        -N*special.gammaln(dof/2) + N*dof*np.log(dof/2)/2 + dof*(np.log(z)-z).sum()/2
    ]

    # iterate
    for i in range(iter):
        # M step
        mean = (X * z[:,None]).sum(axis=0).reshape(-1,1) / z.sum()
        mu = X - mean.squeeze()[None,:]
        cov = np.einsum('ij,ik->jk', mu, mu * z[:,None])/N

        # E step
        delta = (mu * np.linalg.solve(cov,mu.T).T).sum(axis=1)
        delta = np.einsum('ij,ij->i', mu, np.linalg.solve(cov,mu.T).T)
        z = (dof + D) / (dof + delta)

        # store objective
        obj.append(
            -N*np.linalg.slogdet(cov)[1]/2 - (z*delta).sum()/2 \
            -N*special.gammaln(dof/2) + N*dof*np.log(dof/2)/2 + dof*(np.log(z)-z).sum()/2
        )   
        
        if np.abs(obj[-1] - obj[-2]) < eps:
            break
    return cov, mean.squeeze(), obj
#Fitting Face and Non-face to t-model and obtaining the parameters

pca = PCA(n_components=n_components)
trainingFaces=pca.fit_transform(trainingFaces)
trainingNonFaces=pca.fit_transform(trainingNonFaces)
parameters=T_distribution(trainingFaces, dof=4.0, iter=50, eps=1e-1)
parameters1=T_distribution(trainingNonFaces,dof=4.0,iter=50,eps=1e-1)




pca = PCA(n_components=60)

def evaluate_Model(testingface,testingnonface):
    testingface=pca.fit_transform(testingface)
    #testingnonface=pca.fit_transform(testingnonface)
    TP=0
    TN=0
    FP=0
    FN=0
    predictions1=[0]*test_size
    for i in range(0,testsize):
        a=fitting_T(testingface[i],parameters[1],parameters[0],dof,60)
        b=fitting_T(testingface[i],parameters1[1],parameters1[0],dof,60)
        y=np.true_divide(a,(a+b))
        if (y>0.5):
            predictions1.append(1)
            TP=TP+1
    new_TP1=[0]*test_size
    for i in range(len(predictions1)-test_size,len(predictions1)):
        new_TP1[(len(predictions1)-test_size)-i]=predictions1[i]
   
    print ("True Positive of the Model= ",TP)
    FN=testsize-TP
    print ("False Negative of the Model= ",FN)
    
    
    #testingface=pca.fit_transform(testingface)
    testingnonface=pca.fit_transform(testingnonface)
    predictions2=[0]*test_size
    for i in range(0,testsize):
        a=fitting_T(testingnonface[i],parameters[1],parameters[0],dof,60)
        b=fitting_T(testingnonface[i],parameters1[1],parameters1[0],dof,60)
        y=np.true_divide(a,(a+b))
        if (y>0.5):
            predictions2.append(1)
            FP=FP+1
    new_FP1=[0]*test_size
    for i in range(len(predictions2)-test_size,len(predictions2)):
        new_FP1[(len(predictions2)-test_size)-i]=predictions2[i]
    
    print ("False Positive of the model= ",FP)
    TN=testsize-FP
    print ("True Negative of the model= ",TN)
    A=TP+FP
    PRECISION=(np.true_divide(TP,A))*100
    print ("PRECISON OF MODEL = ",PRECISION)
    B=TP+FN
    RECALL=(np.true_divide(TP,B))*100
    C=TP+TN
    D=TP+TN+FP+FN
    ACCURACY=(np.true_divide(C,D))*100
    print ("RECALL OF MODEL = ",RECALL)
    print ("ACCURACY OF MODEL = ",ACCURACY)
    TNFN=TN+FN
    FPR=np.absolute(np.divide(FP,TNFN))
    print ("False Positive Rate = ",FPR)
    FPTP=FP+TP
    FNR=np.absolute(np.divide(FN,FPTP))
    print ("False Negative Rate = ",FNR)
    FPFN=FP+FN
    MCR=np.absolute(np.divide(FPFN,test_size))
    print ("Misclassification Rate = ",MCR)
    plot_ROC(np.asarray(new_TP1),np.asarray(new_FP1),test_size)

def plot_ROC(f_nf, f_f,test_size):
    print ("*************")
    print ("Plotting ROC")
    print ("*************")
    predictions = np.append(f_nf, f_f)
    temp1 = [0]*test_size
    temp2 = [1]*test_size
    actual = np.append(temp1,temp2)
    false_positive_rate, true_positive_rate, _ = roc_curve(actual, predictions)
    roc_auc = auc(false_positive_rate, true_positive_rate)
    
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Model_2 T Distribution')
    plt.legend(loc="lower right")
    plt.plot(false_positive_rate, true_positive_rate, 'b')
    plt.show()
    
evaluate_Model(testingface,testingnonface)