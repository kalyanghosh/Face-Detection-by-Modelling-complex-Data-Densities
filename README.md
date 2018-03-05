# Face-Detection-by-Modelling-complex-Data-Densities
This repository contains code base and the steps followed while performing this project.
<br> </br>
<b>TITLE: FACE DETECTION BY MODELLING COMPLEX DATA DENSITIES</b>
<br></br>
The main objective of this project is to perform face detection by modelling complex data distributions like Single Gaussian Model,
Mixture of Gaussians, T distribution and Factor Analyser
<br></br>

<b>1. GETTING THE DATA : </b>
The first step of this project was to download the face data. For my case I used the University Of Maryland Face dataset available at :
![UMD FACES](http://www.umdfaces.io/").
<br></br>
After downloading the data, I had to write a Python IO module that cropped the faces from the given annotation files and also cropped a   random part from the image as background.For the purpose of creating data for Training & Testing set, I divided the data into 4 folders namely the <b>Train_FaceData</b>,<b>Train_NonFaceData</b>,<b>Test_FaceDate</b>,<b>Test_NonFaceData</b>. The code for this module is attached as Data_Preprocessing_IO_Code.py
<br></br>
<b>2. SINGLE GAUSSIAN MODEL : </b>
<br></br>
In this , I wrote code to model the face data using a Single Gaussian Model.The code for this module is attached as Data_Preprocessing_IO_Code.py. The mean face model and the mean covariance face model learned from this model at shown below.
<br></br>
![MEAN FACE](C:\Users\Kalyan\Desktop\NCSU\2nd Semester\ECE_763_Computer_Vision_&_Deep_Learning\Project1\kghosh_project1\kghosh_project1\Result_Images/Model_1_MeanFace.jpg?raw=true "Title")


