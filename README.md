Distinguish-Handwritten-Alphabet
================================

Introduction
------------
Have you ever had the problem of wanting to convert your handwritting into text on your devices? Well, I have and it is frustrating and time consuming to copy texts manually constently. Software Engineers and many machine learning researchers have came up with lots of ideas on how to implement this task, and here I decided to tackle this problem using Neural Networks.


Installing Packages
-------------------
Packages used:

```python
import warnings
import pandas as pd
import matplotlib.pyplot as plt
import sklearn
import os
import cv2 
import csv
```

Installing package, run:

```terminal
pip install pandas, matplotlib, sklearn, os, cv2, csv, warnings
```

Quick Overview
--------------
In order to distinguish handwritten alphabets, I started off planning the machine learning model. The whole process can be separated through a pipeline (as seen down below). The first part is gathering enough images to be represented as data, then converting the image into useful data for the neural network, then followed by training the neural network to be able to predict new photos


Initialising Data
-----------------














