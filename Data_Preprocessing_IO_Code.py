# -*- coding: utf-8 -*-
"""
Created on Tue Feb 13 21:05:49 2018

@author: Kalyan
"""

import os
import xml.etree.ElementTree as ET
import cv2
import glob
import numpy as np
#import pandas as pd
import csv
#Defining the constants (Image Dimensions,cropping window size)
x0=0
x1=60
y0=60
y1=60
x_dim=60
y_dim=60

data_path='C:\\Users\\Kalyan\\Desktop\\CV_Project_763\\umd_dataset\\test_images\\'
image_path='C:\\Users\\Kalyan\\Desktop\\CV_Project_763\\umd_dataset\\test_images\\'
cropped_path='C:\\Users\\Kalyan\\Desktop\\CV_Project_763\\umd_dataset\\test_images\\'
annotation_file='C:\\Users\\Kalyan\\Desktop\\CV_Project_763\\umd_dataset\\images\\umdfaces_batch3_ultraface.csv'

def dataProcess(data_path,threshold):
    threshold_val=threshold
    image=glob.glob(image_path+"*.jpg")
    for img in image:
        #print (img)
        list1=img.split('\\')
        imageName=list1[len(list1)-1]
        img1=cv2.imread(img)
        img2=cv2.imread(img)
        with open (annotation_file,'r') as csvfile:
            content=csv.reader(csvfile)
            for row in content:
                
                name=row[1]
                if(imageName in name):
                    print ("Found")
                    #Taking care of the face
                    X1=(row[4]).split('.')
                    Y1=(row[5]).split('.')
                    W1=(row[6]).split('.')
                    H1=(row[7]).split('.')
                    X=int(X1[0])
                    Y=int(Y1[0])
                    W=int(W1[0])
                    H=int(H1[0])
                    print (X,Y,W,H)
                    cv2.rectangle(img1,(X,Y),(X+H,Y+H),(255,255,255))
                    cropped_image=img1[Y:Y+H,X:X+W]
                    gray=cv2.cvtColor(cropped_image,cv2.COLOR_RGB2GRAY)
                    resized=cv2.resize(gray,(x_dim,y_dim))           
                    #fname,ext=os.path.splitext(img)
                    #cv2.imwrite(os.path.join(image_path,fname+"_resized"+ext),resized)
                    #Taking care of the background
                    back=img2[0:x_dim,0:y_dim]
                    back_gray=cv2.cvtColor(back,cv2.COLOR_RGB2GRAY)
                    #fname,ext=os.path.splitext(img)
                    #cv2.imwrite(os.path.join(image_path,fname+"_background"+ext),back_gray)
                    #Calculating the correlation,coffecient
                    res = cv2.matchTemplate(resized,back_gray,cv2.TM_CCORR_NORMED)
                    t= (res[0][0])
                    print (t)
                    if(t>=threshold_val):
                        fname1,ext=os.path.splitext(img)
                        cv2.imwrite(os.path.join(image_path,fname1+"_resized"+ext),resized)
                        fname2,ext=os.path.splitext(img)
                        cv2.imwrite(os.path.join(image_path,fname2+"_background"+ext),back_gray)
                else:
                    continue
    return
               

dataProcess(data_path,0.7)
        
        
        

    
