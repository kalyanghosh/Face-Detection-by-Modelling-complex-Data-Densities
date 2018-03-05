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
In this , I wrote code to model the face data using a Single Gaussian Model.The code for this module is attached as <b>Data_Preprocessing_IO_Code.py</b>. The mean face model and the mean covariance face model learned from this model at shown below.
<br></br>
<br><b>MEAN FACE:</b></br>
![MEAN FACE](https://github.com/kalyanghosh/Face-Detection-by-Modelling-complex-Data-Densities/blob/master/Model_1_MeanFace.png)
<br><b>MEAN COVARIANCE FACE:</b></br>
![MEAN FACE](https://github.com/kalyanghosh/Face-Detection-by-Modelling-complex-Data-Densities/blob/master/Model_1_Covariance_Face.png)
<br><b>ROC:</b></br>
![MEAN FACE](https://github.com/kalyanghosh/Face-Detection-by-Modelling-complex-Data-Densities/blob/master/ROC_Model_1_Gaussian.png)
<br></br>
<b>2. MIXTURE OF GAUSSIAN : </b>
<br></br>
In this , I wrote code to model the face data using a Mixture of Gaussian for K=3.The code for this module is attached as <b>Model_2_Mixture_Of_Gaussians.py</b>. Here, I used the <b>Expectation-Maximization</b> algorithm to estimate the parameters of the model. The mean face model and the mean covariance face model learned from this model at shown below.
<br></br>
<br><b>MEAN FACE:</b></br>
![MEAN FACE](https://github.com/kalyanghosh/Face-Detection-by-Modelling-complex-Data-Densities/blob/master/Model_2_MeanFace.png)
<br><b>MEAN COVARIANCE FACE:</b></br>
![MEAN FACE](https://github.com/kalyanghosh/Face-Detection-by-Modelling-complex-Data-Densities/blob/master/Model_1_Covariance_Face.png)
<b>3. T DISTRIBUTION : </b>
<br></br>
In this , I wrote code to model the face data using a T distribution.The code for this module is attached as <b>Model_3_T.py</b>. Here, I used the <b>Expectation-Maximization</b> algorithm to estimate the parameters of the model. The mean face model and the mean covariance face model learned from this model at shown below.
<br></br>
<br><b>MEAN FACE:</b></br>
![MEAN FACE](https://github.com/kalyanghosh/Face-Detection-by-Modelling-complex-Data-Densities/blob/master/Model_2_MeanFace.png)
<br><b>MEAN COVARIANCE FACE:</b></br>
![MEAN FACE](https://github.com/kalyanghosh/Face-Detection-by-Modelling-complex-Data-Densities/blob/master/Model_1_Covariance_Face.png)
<b>4. FACTOR ANALYZER : </b>
<br></br>
In this , I wrote code to model the face data using a Factor Analyzer.The code for this module is attached as <b>Model_5_Factor_Analyser.py</b>. Here, I used the <b>Expectation-Maximization</b> algorithm to estimate the parameters of the model. The mean face model and the mean covariance face model learned from this model at shown below.
<br></br>
<br><b>MEAN FACE:</b></br>
![MEAN FACE](https://github.com/kalyanghosh/Face-Detection-by-Modelling-complex-Data-Densities/blob/master/Model_5_MeanFace.png)
<br><b>MEAN COVARIANCE FACE:</b></br>
![MEAN FACE](https://github.com/kalyanghosh/Face-Detection-by-Modelling-complex-Data-Densities/blob/master/Model_4_Covariance_Face.png)
<br><b>ROC:</b></br>
![MEAN FACE](https://github.com/kalyanghosh/Face-Detection-by-Modelling-complex-Data-Densities/blob/master/Model_5_ROC.png)
<br></br>

