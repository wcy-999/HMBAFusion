# HMBAFusion
This is the official implementation of the paper submitted to The Visual Computer.

## Introduction
This repository provides the code for HMBAFusion, a hierarchical multi-branch attention framework designed for high-fidelity medical image integration. Our method effectively captures global context, regional patterns, and local visual cues to support advanced visual computing tasks such as 3D medical volume rendering and surgical navigation.

## Platform
Python 3.7  
torch >=1.0  


## Datasets

The training datasets and test datasets are built on the publicly available Harvard medical dataset(http://www.med.harvard.edu/AANLIB/home.html). 
(1) For SPECT-MRI image fusion experiments, we selected 333 pairs of images as the training set and 24 pairs as the test set. 
(2) For PET-MRI image fusion experiments, we selected 245 pairs of images as the training set and 24 pairs as the test set. 
(3) For CT-MRI image fusion experiments, we selected 160 pairs of images as the training set and 24 pairs as the test set.


If you have any questions about the code, feel free to contact me at e22301219@stu.ahu.edu.cn.
