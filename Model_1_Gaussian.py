# -*- coding: utf-8 -*-
"""
Created on Sat Feb 17 18:39:58 2018

@author: Kalyan
"""

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
from sklearn.metrics import roc_curve, auc

#Code for Data Modelling
TrainFacePath='C:\\Users\\kghosh\\Downloads\\Latest_CV_Codes_28th\\Latest_CV_Codes_28th\\CV_Project_763\\Train_FaceData\\'
TrainNonFacePath='C:\\Users\\kghosh\\Downloads\\Latest_CV_Codes_28th\\Latest_CV_Codes_28th\\CV_Project_763\\Train_NonFaceData\\'
TestFacePath='C:\\Users\\kghosh\\Downloads\\Latest_CV_Codes_28th\\Latest_CV_Codes_28th\\CV_Project_763\\Test_FaceData\\'
TestNonFacePath='C:\\Users\\kghosh\\Downloads\\Latest_CV_Codes_28th\\Latest_CV_Codes_28th\\CV_Project_763\\Test_NonFaceData\\'
train_size=1000
test_size=100
dim=3600
number_of_components=100
threshold=0.5
height=60
width=60


        
def mean(data):
        mean_matrix=data.mean(1)
        return mean_matrix

def mean_show(data):
    mean_matrix=data.mean(0)
    mean_image=np.reshape(mean_matrix,(height, width))
    plt.imshow(mean_image)
    
    
    

def covariance(data):
        E=np.zeros(shape=(number_of_components,number_of_components))
        covariance=np.cov(data)
        for i in range(len(covariance)):
            for j in range(len(covariance)):
                if(i==j):
                    E[i][j]=covariance[i][j]
        return E
    

def gaussian(mean_matrix,covariace_matrix,data):
        a=(data-mean_matrix)
        a_T=np.transpose(a)
        E_inv=linalg.inv(covariace_matrix)
        c=np.matmul(a_T,E_inv)
        d=np.matmul(c,a)
        e=np.exp(-d)
        return e

    

def perform_PCA(data,number_of_components):
        #print ("Data",data.shape) 
        pca=PCA(n_components=number_of_components)
        #print (pca.fit_transform(data).shape)
        return (pca.fit_transform(data).transpose())
   
def read_Data(filepath):  
        image=glob.glob(filepath+"*.jpg")
        X = []
        for img in image:
            img=cv2.imread(img)
            #img=cv2.normalize(img,None,alpha=0,beta=1,norm_type=cv2.NORM_MINMAX,dtype=cv2.CV_32F)
            img=cv2.cvtColor(img,cv2.COLOR_RGB2GRAY)
            img=img.flatten().tolist()
            X.append(img)
        
        X=np.matrix(X)
        X=np.array(X)
        return X


def train_face(TrainFacePath):
        dataMatrix=read_Data(TrainFacePath)
        mean_show(dataMatrix)
        pca_data=perform_PCA(dataMatrix,number_of_components)
        mu=mean(pca_data)
        cov=covariance(pca_data)
        return (mu,cov)

def train_nonface(TrainNonFacePath):
        dataMatrix=read_Data(TrainNonFacePath)
        #mean_show(dataMatrix)
        pca_data=perform_PCA(dataMatrix,number_of_components)
        mu=mean(pca_data)
        cov=covariance(pca_data)
        return (mu,cov)
    
    
    
    

def test_face(TestFacePath):
        dataMatrix=read_Data(TestFacePath)
        pca_data=perform_PCA(dataMatrix,number_of_components)
        mu_face,E_face=train_face(TrainFacePath)
        mu_nonface,E_nonface=train_nonface(TrainNonFacePath)
        result=[None]*len(pca_data)
        for i in range(0,len(pca_data)):
            x=pca_data[:,i]
            a=gaussian(mu_face,E_face,x)
            b=gaussian(mu_nonface,E_nonface,x)
            res=(a/a+b)
            if(res<=threshold):
                result[i]=float(0)
            else:
                result[i]=float(1)
        return (result)
        
        
def test_nonface(TestNonFacePath):
        dataMatrix=read_Data(TestNonFacePath)
        pca_data=perform_PCA(dataMatrix,number_of_components)
        mu_face,E_face=train_face(TrainFacePath)
        mu_nonface,E_nonface=train_nonface(TrainNonFacePath)
        result=[0 for i in range(0,len(pca_data))]
        for i in range(0,len(pca_data)):
            x=pca_data[:,i]
            a=gaussian(mu_face,E_face,x)
            b=gaussian(mu_nonface,E_nonface,x)
            res=(b/(a+b))
        
            if(res<=threshold):
                result[i]=0
            else:
                result[i]=1
        return (result)

def evaluate_model():
       F=test_face(TestFacePath)
       NF=test_nonface(TestNonFacePath)
       TP=F.count(1)
       TN=NF.count(1)
       FP=NF.count(0)
       FN=F.count(0)
       a=TP+FP
       PRECISION=(np.true_divide(TP,a))*100
       b=TP+FN
       RECALL=(np.true_divide(TP,b))*100
       c=TP+TN
       d=TP+TN+FP+FN
       ACCURACY=(np.true_divide(c,d))*100
       print ("*************")
       print ("PRECISON OF MODEL = ",PRECISION)
       #print ("RECALL OF MODEL = ",RECALL)
       print ("ACCURACY OF MODEL = ",ACCURACY)
       print ("*************")
       FPR=np.absolute(np.divide(FP,np.sum(TN,FN)))
       print ("False Positive Rate = ",FPR)
       #print (TP,TN,FP,FN)
       FNR=np.absolute(np.divide(FN,(FP+TP)))
       print ("False Negative Rate = ",FNR)
       FPFN=FP+FN
       MCR=np.absolute(np.divide(FPFN,test_size))
       print ("Misclassification Rate = ",MCR)
       plot_ROC(NF, F,test_size)

def plots(TrainFacePath,TrainNonFacePath,TestFacePath,TestNonFacePath):
     data=read_Data(TrainNonFacePath)
     mean_matrix=data.mean(0)
     mean_image=np.reshape(mean_matrix,(height, width))
     #plt.imshow(mean_image)
     covariance=np.cov(data.T)
     covariance=covariance.mean(0)
     cov_image=np.reshape(covariance,(height, width))
     plt.imshow(cov_image)
   
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
    plt.title('ROC Model_1 Gaussian')
    plt.legend(loc="lower right")
    plt.plot(false_positive_rate, true_positive_rate, 'b')
    plt.show()
    
#evaluate_model()
#img1=read_Data(TrainFacePath)
#mean_img=mean_show(img1)
#imgplot = plt.imshow(mean_img,cmap='gray')

plots(TrainFacePath,TrainNonFacePath,TestFacePath,TestNonFacePath)
   






