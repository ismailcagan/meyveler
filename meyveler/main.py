import json
import os

from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Activation, Dropout, Flatten, Dense
from keras.preprocessing.image import ImageDataGenerator, img_to_array, load_img
import matplotlib.pyplot as plt
from glob import glob
import numpy as np


train_path=("meyve-dataset\\fruits-360\\Training\\")
test_path=("meyve-dataset\\fruits-360\\Test\\")

img=load_img(train_path+ "Apple Braeburn\\0_100.jpg") # resim denedik
#img1=os.listdir(train_path+ "Apple Braeburn\\0_100.jpg")
plt.imshow(img)
plt.axis("off")
plt.show()


x=img_to_array(img) # resmi diziye çevirdik, ileriye doğru boyutu lazım olacak (100,100,3)
print("görüntü boyutu :",x.shape)


ClassName=glob(train_path+"\\*") # train içindeki bütün resimleri çekiyoruz
print(ClassName)   # class isimlerine bakıyoruz
numberofclass=len(ClassName) # kaç tane clas olduğunu bakıyoruz 131 farklı meyve var
print("number of class :",numberofclass)




model=Sequential()

model.add(Conv2D(32,(5,5),input_shape=x.shape))
model.add(Activation("relu"))
model.add(MaxPooling2D())

model.add(Conv2D(32,(5,5)))
model.add(Activation("relu"))
model.add(MaxPooling2D())


model.add(Conv2D(64,(3,3)))
model.add(Activation("relu"))
model.add(MaxPooling2D())

model.add(Flatten())
model.add(Dense(1024))
model.add(Activation("relu"))
model.add(Dropout(0.5))
model.add(Dense(numberofclass)) #output layer class sayısı
model.add(Activation("softmax"))


model.compile(loss="categorical_crossentropy",optimizer="rmsprop",metrics=["accuracy"])
batch_size=32
model.summary()


train_datagen=ImageDataGenerator(
    rescale=1./255,             # normalize ediyouz
    shear_range=0.3,            # belli bir açıyla dönme yapıyor
    horizontal_flip=True,       # random şeklinde sağa ve sola çevrilecek
    zoom_range=0.3              # # belli  bir oranda yakınlaşacak
)


test_datagen=ImageDataGenerator(
    rescale=1./255,
    shear_range=0.3,
    horizontal_flip=True,
    zoom_range=0.3
)

#history=model.fit_generator(datagen.flow(x_train,y_train, batch_size=batch_size),epochs=epochs, validation_data=(x_val, y_val), steps_per_epoch=10)#x_train.shape[0]//batch_size)
train_generator=train_datagen.flow_from_directory(

train_path,                 # train içinde bulunan meyveler
target_size=x.shape[:2],    # meyvelerin boyutu (100,100)
batch_size=batch_size,      # batch size
color_mode="rgb",           # renk kanalı
class_mode="categorical")   # class modu



test_generator=test_datagen.flow_from_directory(

test_path,
target_size=x.shape[:2],
batch_size=batch_size,
color_mode="rgb",
class_mode="categorical")

#history=model.fit_generator(datagen.flow(x_train,y_train, batch_size=batch_size),epochs=epochs, validation_data=(x_val, y_val), steps_per_epoch=10)#x_train.shape[0]//batch_size)
"""
hist=model.fit_generator(
    generator=train_generator,
    steps_per_epoch=1600 // batch_size,
    epochs=100,
    validation_data=test_generator,
    validation_steps=800//batch_size) # step per epochla aynı mantık , veri yaratıyor

score = model.evaluate(train_generator, test_generator, verbose=0)
print('Test loss:', score[0])
print('Test accuracy:', score[1])
"""

#score = model.evaluate(x_train, y_train, verbose=0)
#print('Test loss:', score[0])
#print('Test accuracy:', score[1])



import tensorflow as tf

from tensorflow.python.client import device_lib
print(device_lib.list_local_devices())

print("$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$")
print("GPU Kullanımda (hayır=0 , evet=1): ", len(tf.config.experimental.list_physical_devices('GPU')))

print("$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$")
import tensorflow as tf
Sonuc = tf.test.is_gpu_available(
cuda_only=False,
min_cuda_compute_capability=None
)
print(Sonuc)












"""
#model save (modeli deneme ve kaydetme)

model.save_weights("deneme.h5")

# model evaluation
print(hist.history.keys())
plt.plot(hist.history["loss"],label="train loss")
plt.plot(hist.history["val_loss"],label="validation loss")
plt.legend()
plt.show()

plt.figure()

plt.plot(hist.history["acc"],label="train acc")
plt.plot(hist.history["val_acc"],label="validation acc")
plt.legend()
plt.show()
"""

"""
# save history
import json
with open("deneme.json","w") as f: # daha önce kaydetilen dosyayı içine koyuyoruz
    json.dump(hist.history,f)
"""

"""
# load history

import codecs
with codecs.open("cnn_fruit_hist.json","r",encoding="utf-8") as f:
    h=json.loads((f.read()))

plt.plot(h["loss"],label="train loss")
plt.plot(h["val_loss"],label="validation loss")
plt.legend()
plt.show()

plt.figure()

plt.plot(h["acc"],label="train acc")
plt.plot(h["val_acc"],label="validation acc")
plt.legend()
plt.show()

"""








