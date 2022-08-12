# Project 1 
![](https://github.com/thatssweety/Images/blob/7655044de3a0b236f70f110aa67bfcc78337879f/you%20(2).png)
 # Data : 1000 images for each class (13 classes in total)
 ![](https://github.com/thatssweety/Images/blob/e632b15a8f7de6e566ce51530c42cc2c4c71772b/Image%20Data%20fot%20training%20and%20validation.png)
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



