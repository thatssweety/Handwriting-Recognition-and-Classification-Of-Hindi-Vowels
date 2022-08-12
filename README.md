# Project 1 : Handwriting-Recognition-and-Classification-Of-Hindi-Vowels

![](you (2).png)
 # Data : 1000 images for each class (13 classes in total)
 ![](Image Data fot training and validation.png)
 # Code
  ```
  import numpy as np
import matplotlib.pyplot as plt
import os
import cv2
from tqdm import tqdm

DATADIR = "/content/NEW DATASET"

CATEGORIES = ["A","AA","AE","AH","ANG","AYE","E","EE","EO","O","OO","RI","U"]

for category in CATEGORIES: 
    path = os.path.join(DATADIR,category)  
    for img in os.listdir(path):  # iterate over each image 
        img_array = cv2.imread(os.path.join(path,img) ,cv2.IMREAD_GRAYSCALE) 
        img_array=cv2.bitwise_not(img_array) # convert to array
        plt.imshow(img_array, cmap='gray')  # graph 
        plt.show()  # display

        break  # we just want one for now so break
       
       ```
