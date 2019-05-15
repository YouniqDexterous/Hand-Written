import numpy as np
import pandas as pd
import seaborn as sns
# import matplotlib.pyplot as plt
from sklearn.externals import joblib


import warnings
warnings.filterwarnings('ignore')

train = pd.read_csv("/Users/yogesh/Desktop/CollegeProject/CV_proj/cv_oct/train.csv")
test= pd.read_csv("/Users/yogesh/Desktop/CollegeProject/CV_proj/cv_oct/test.csv")

Y_train = train["label"]
X_train = train.drop(labels = ["label"],axis = 1)

X_train = X_train.astype('float16') / 255.0
test = test.astype('float16') / 255.0

X_train = X_train.values.reshape(-1,28,28,1)
test = test.values.reshape(-1,28,28,1)

from sklearn.model_selection import train_test_split
X_train, X_val, Y_train, Y_val = train_test_split(X_train, Y_train, test_size = 0.1, random_state=2)

from keras.utils.np_utils import to_categorical
Y_train = to_categorical(Y_train, num_classes = 10)

from sklearn.metrics import confusion_matrix
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPool2D
from keras.optimizers import RMSprop,Adam
from keras.preprocessing.image import ImageDataGenerator


model = Sequential()
#
model.add(Conv2D(filters = 8, kernel_size = (5,5),padding = 'Same',
                 activation ='relu', input_shape = (28,28,1)))
model.add(MaxPool2D(pool_size=(2,2)))
model.add(Dropout(0.25))
#
model.add(Conv2D(filters = 16, kernel_size = (3,3),padding = 'Same',
                 activation ='relu'))
model.add(MaxPool2D(pool_size=(2,2), strides=(2,2)))
model.add(Dropout(0.25))

model.add(Flatten())
model.add(Dense(256, activation = "relu"))
model.add(Dropout(0.5))
model.add(Dense(10, activation = "softmax"))

optimizer = Adam(lr=0.001, beta_1=0.9, beta_2=0.999)

model.compile(optimizer = optimizer , loss = "categorical_crossentropy", metrics=["accuracy"])


epochs = 10
batch_size = 250

datagen = ImageDataGenerator(
        featurewise_center=False,
        samplewise_center=False,
        featurewise_std_normalization=False,
        samplewise_std_normalization=False,
        zca_whitening=False,
        rotation_range=5,
        zoom_range = 0.05,
        width_shift_range=0.05,
        height_shift_range=0.05,
        horizontal_flip=False,
        vertical_flip=False)

datagen.fit(X_train)

history = model.fit_generator(datagen.flow(X_train,Y_train, batch_size=batch_size),
                              epochs=epochs,steps_per_epoch=37800)
import seaborn as sns


Y_pred = model.predict(X_val)
#
Y_pred_classes = np.argmax(Y_pred,axis = 1)
#
Y_true = np.argmax(Y_val,axis = 1)
# confusion_mtx = confusion_matrix(Y_true, Y_pred_classes)
# f,ax = plt.subplots(figsize=(8, 8))
# sns.heatmap(confusion_mtx, annot=True, linewidths=0.01,cmap="Greens",linecolor="gray", fmt= '.1f',ax=ax)
# plt.xlabel("Predicted Label")
# plt.ylabel("True Label")
# plt.title("Confusion Matrix")
# plt.show()

# output_label = model.predict(test)
# output_label1=np.argmax(output_label,axis=1)
# output = pd.DataFrame(output_label1,columns = ['Label'])
# output.reset_index(inplace=True)
# output['index'] = output['index'] + 1
# output.rename(columns={'index': 'ImageId'}, inplace=True)
# output.to_csv('output.csv', index=False)

f='model.pkl'
joblib.dump(model,f)
