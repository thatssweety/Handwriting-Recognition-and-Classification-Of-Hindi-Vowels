# **Project 1** 
![alt text](https://github.com/thatssweety/Images/blob/7655044de3a0b236f70f110aa67bfcc78337879f/you%20(2).png?raw=true)


# 1000 images for each class (13 classes in total)
 ![alt text](https://github.com/thatssweety/Images/blob/efcd70d1f839a3dea9b10169c92c5790988f0e67/Screenshot%20(434).png?raw=true)

#  **Tech Stack Used** :

1. Numpy
2. Pandas
3. Tensorflow
4. OpenCV
5. Keras

# Code
 In Python
 
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
# Creating Training Data using images

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
![alt text](https://github.com/thatssweety/Images/blob/8c915ddf5cbc49274668af7e61e671d160f5dd12/Screenshot%20(444)%20-%20Copy.png?raw=true)


# Creating features and label and saving them

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
![alt text](https://github.com/thatssweety/Images/blob/94a9b3977f7e98fc190522c6c0580d2f3439e152/Screenshot%20(440).png?raw=true)

# Saving Model
```
from tensorflow.keras.models import Sequential, save_model, load_model
filepath = './saved_model'
save_model(model, filepath)
model = load_model(filepath, compile = True)
```
# Making Predictions

```
img_array = cv2.imread('/content/gdrive/MyDrive/mnist/A.jpg',cv2.IMREAD_GRAYSCALE)            
plt.imshow(img_array,cmap='gray')
plt.show()
IMG_SIZE=50
new_array = cv2.resize(img_array, (IMG_SIZE, IMG_SIZE))
new_array=cv2.bitwise_not(new_array)
#new_array=cv2.bitwise_not(new_array)
plt.imshow(new_array,cmap='gray')
plt.show()
new_array=np.array(new_array).reshape(-1, IMG_SIZE, IMG_SIZE, 1)
samples_to_predict = []
samples_to_predict.append(new_array)
```
![alt text](https://github.com/thatssweety/Images/blob/94e646ab11d7acde88fb8bfae25272a4b3abbc33/Screenshot%20(441).png?raw=true)

# Predictions

```
predictions = model.predict(new_array)
print(predictions)
classes = np.argmax(predictions, axis = 1)
print(classes)

```
![alt text](https://github.com/thatssweety/Images/blob/47bd0051dcedeadd30d85d77fb970f569e293511/Screenshot%20(443).png?raw=true)

