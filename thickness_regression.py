import keras,os
from keras.models import Sequential
from keras.layers import Dense, Conv2D, MaxPool2D , Flatten
from keras.preprocessing.image import ImageDataGenerator
import numpy as np
from tensorflow.keras.applications.vgg16 import VGG16

from tensorflow.keras.applications.inception_v3 import InceptionV3
from tensorflow.keras.applications.densenet import DenseNet121
from tensorflow.keras.applications.resnet import ResNet50
from tensorflow.keras.applications.xception import Xception
from tensorflow.keras.applications.nasnet import NASNetLarge
from tensorflow.keras.applications.mobilenet_v2 import MobileNetV2
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D

from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
import matplotlib.pyplot as plt
from tensorflow.keras.callbacks import ModelCheckpoint

import numpy as np
from PIL import Image
from os import *
from os.path import *
import cv2
import tensorflow as tf
import datetime
import data_loader

epochs = 100
max_labels = 27 ### number of classes = number of layers
img_ext = '.png'
txt_ext = '.txt'
np_ext = '.npy'
models = 'models'
plots = 'plots'
evaluation = 'evaluation'
history = 'history'
out = 'out'
timestamp = datetime.datetime.now()
dt_format = "%m-%d-%y_%H-%M"
models = join(out, timestamp.strftime(dt_format),models)
plots = join(out, timestamp.strftime(dt_format),plots)
evaluation = join(out, timestamp.strftime(dt_format),evaluation)
history = join(out, timestamp.strftime(dt_format),history)

if not isdir(models):
    makedirs(models)
if not isdir(plots):
    makedirs(plots)
if not isdir(evaluation):
    makedirs(evaluation)
if not isdir(history):
    makedirs(history)
    
img_root = '../../Dataset/Snow Radar/2012_cropped/train/image/'
thick_root = '../../Dataset/Snow Radar/2012_cropped/train/thickness_estimates2/'

def train_model(base_model, model_name, traindata, thickness_estims):
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(1024, activation='relu')(x)
    predictions = Dense(max_labels, activation='relu')(x)
    model = Model(inputs=base_model.input, outputs=predictions)
    
    model_chkpnt = ModelCheckpoint(
    filepath=join(models,model_name+'.h5'),
    save_weights_only=False,
    monitor='loss',
    mode='min',
    save_best_only=True)
    
    model.compile(optimizer=Adam(lr=0.0001), loss='mae', metrics=['accuracy'])
    
    hist = model.fit(traindata, thick_estims, verbose=1, epochs=epochs, callbacks=[model_chkpnt])
    print("-- " + model_name + ' trained -- ')
    np.save(join(history,model_name+np_ext),hist.history)
    np.savetxt(join(evaluation,model_name+txt_ext), model.evaluate(traindata, thick_estims, verbose=1))
    return hist

def plot_model(hist, model_name, folder_out=plots):
    print(hist.history.keys())
    plt.plot(hist.history["loss"])
    plt.title(model_name)
    plt.ylabel("MAE Loss")
    plt.xlabel("Epoch")
    plt.yscale("linear")
    plt.savefig(join(folder_out,model_name+img_ext))
    plt.close()


traindata = data_loader.traindata


thick_estims = data_loader.train_thick

### model definition ###

model_name1 = 'InceptionV3'
print('training ' + model_name1)
inception = InceptionV3(weights='imagenet', include_top=False)
hist1 = train_model(inception, model_name1, traindata, thick_estims)
plot_model(hist1, model_name1)


model_name2 = 'DenseNet121'
print('training ' + model_name2)
densenet = DenseNet121(weights='imagenet', include_top=False)
hist2 = train_model(densenet, model_name2, traindata, thick_estims)
plot_model(hist2, model_name2)


model_name3 = 'ResNet50'
print('training ' + model_name3)
resnet = ResNet50(weights='imagenet', include_top=False)
hist3 = train_model(resnet, model_name3, traindata, thick_estims)
plot_model(hist3, model_name3)


model_name4 = 'Xception'
print('training ' + model_name4)
xception = Xception(weights='imagenet', include_top=False)
hist4 = train_model(xception, model_name4, traindata, thick_estims)
plot_model(hist4, model_name4)


model_name5 = 'MobileNetV2'
print('training ' + model_name5)
mobilenet = MobileNetV2(weights='imagenet', include_top=False)
hist5 = train_model(mobilenet, model_name5, traindata, thick_estims)
plot_model(hist5, model_name5)


## plot all
plt.plot(hist1.history["loss"])
plt.plot(hist2.history["loss"])
plt.plot(hist3.history["loss"])
plt.plot(hist4.history["loss"])
plt.plot(hist5.history["loss"])
plt.title("Model Loss Curves")
plt.ylabel("MAE Loss")
plt.xlabel("Epoch")
plt.yscale("linear")
plt.legend([model_name1,model_name2,model_name3,model_name4,model_name5])
plt.savefig(join(plots,'all_loss'+img_ext))
plt.close()

print('done')