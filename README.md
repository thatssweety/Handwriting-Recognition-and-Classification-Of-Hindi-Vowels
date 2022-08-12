# Project 1 
![](https://github.com/thatssweety/Images/blob/7655044de3a0b236f70f110aa67bfcc78337879f/you%20(2).png)
###    
###
##

## 1000 images for each class (13 classes in total)
 ![](https://github.com/thatssweety/Images/blob/efcd70d1f839a3dea9b10169c92c5790988f0e67/Screenshot%20(434).png)

## Tech Stack Used :
1. Numpy
2. Pandas
3. Tensorflow
4. OpenCV
5. Keras

 # Code
 
 # Loading data and Image Pre-Processing
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
       
IMG_SIZE=50
new_array = cv2.resize(img_array, (IMG_SIZE, IMG_SIZE))
plt.show()

```
### Creating Training Data using images

```
training_data = []

def create_training_data():
    for category in CATEGORIES:  

        path = os.path.join(DATADIR,category)  
        class_num = CATEGORIES.index(category) 
        for img in tqdm(os.listdir(path)):  
            try:
                img_array = cv2.imread(os.path.join(path,img) ,cv2.IMREAD_GRAYSCALE)  # convert to array
                new_array = cv2.resize(img_array, (IMG_SIZE, IMG_SIZE))
                new_array=cv2.bitwise_not(new_array) 
              
                #change black to white
                training_data.append([new_array, class_num])  # add this to our training_data
            except Exception as e:  # in the interest in keeping the output clean...
                pass
create_training_data()

```
### Creating features and label and saving them

```
import random
random.shuffle(training_data)
X = []
y = []
for features,label in training_data:
    X.append(features)
    y.append(label)
print(X[0].reshape(-1, IMG_SIZE, IMG_SIZE, 1))
X = np.array(X).reshape(-1, IMG_SIZE, IMG_SIZE, 1)
import pickle
pickle_out = open("X.pickle","wb")
pickle.dump(X, pickle_out)
pickle_out.close()
pickle_out = open("y.pickle","wb")
pickle.dump(y, pickle_out)
pickle_out.close()
import pickle
pickle_in = open("X.pickle","rb")
X = pickle.load(pickle_in)
pickle_in = open("y.pickle","rb")
y = pickle.load(pickle_in)

```
# CNN MODEL
```
import tensorflow as tf
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten
from tensorflow.keras.layers import Conv2D, MaxPooling2D
X = X/255.0
y=np.array(y)

model = Sequential()

model.add(Conv2D(16, (3, 3), input_shape=X.shape[1:]))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(16, (3, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Flatten())  # this converts our 3D feature maps to 1D feature vectors

model.add(Dense(32))

model.add(Dense(13))
model.add(Activation('softmax'))

model.compile(loss='sparse_categorical_crossentropy',
              optimizer='adam',
              metrics=['sparse_categorical_accuracy'])

model.fit(X, y, batch_size=32, epochs=3, validation_split=0.4)
```



