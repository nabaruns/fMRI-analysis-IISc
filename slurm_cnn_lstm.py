import os
import argparse
import shutil
import subprocess
import json
import warnings
import sys, glob

from multiprocessing import Pool

import numpy as np



in_dir = sys.argv[1]

from tweaked_ImageGenerator_v2 import ImageDataGenerator
#tf version 1.14.0
datagen = ImageDataGenerator()

# some global params
SIZE = (91, 109, 91)
CHANNELS = 1
NBFRAME = 20

# load and iterate training dataset
train_it = datagen.flow_from_directory(os.path.join(in_dir,'train'), target_size=(160, 91, 109, 91), batch_size=1, color_mode='grayscale', frames_per_step=NBFRAME)
# load and iterate validation dataset
val_it = datagen.flow_from_directory(os.path.join(in_dir,'val'), target_size=(160, 91, 109, 91), batch_size=1, color_mode='grayscale', frames_per_step=NBFRAME)
# load and iterate test dataset
test_it = datagen.flow_from_directory(os.path.join(in_dir,'test'), target_size=(160, 91, 109, 91), batch_size=1, color_mode='grayscale', frames_per_step=NBFRAME)

batchX, batchy = train_it.next()
print('Batch shape=',str(batchX.shape),str(batchy.shape))

from keras.layers import Conv3D, BatchNormalization, \
    MaxPool3D, GlobalMaxPool3D

def build_convnet(shape=(112, 112, 112, 3)):
    momentum = .9
    model = keras.Sequential()
    model.add(Conv3D(64, (3,3,3), input_shape=shape,
        padding='same', activation='relu'))
    model.add(Conv3D(64, (3,3,3), padding='same', activation='relu'))
    model.add(BatchNormalization(momentum=momentum))
    
    model.add(MaxPool3D())
    
    model.add(Conv3D(128, (3,3,3), padding='same', activation='relu'))
    model.add(Conv3D(128, (3,3,3), padding='same', activation='relu'))
    model.add(BatchNormalization(momentum=momentum))
    
    model.add(MaxPool3D())
    
    model.add(Conv3D(256, (3,3,3), padding='same', activation='relu'))
    model.add(Conv3D(256, (3,3,3), padding='same', activation='relu'))
    model.add(BatchNormalization(momentum=momentum))
    
    model.add(MaxPool3D())
    
    model.add(Conv3D(512, (3,3,3), padding='same', activation='relu'))
    model.add(Conv3D(512, (3,3,3), padding='same', activation='relu'))
    model.add(BatchNormalization(momentum=momentum))
    
    # flatten...
    model.add(GlobalMaxPool3D())
    return model

from keras.layers import TimeDistributed, GRU, Dense, Dropout, LSTM
def action_model(shape=(160, 91, 109, 91, 1), nbout=3):
    # Create our convnet with (112, 112, 3) input shape
    convnet = build_convnet(shape[1:])
    
    # then create our final model
    model = keras.Sequential()    
    model.add(TimeDistributed(convnet, input_shape=shape))    
    model.add(LSTM(1024, activation='relu', return_sequences=False))
    model.add(Dense(1024, activation='relu'))
    model.add(Dropout(.5))
    model.add(Dense(512, activation='relu'))
    model.add(Dropout(.5))
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(.5))
    model.add(Dense(64, activation='relu'))
    model.add(Dense(nbout, activation='softmax'))
    return model

import glob,os, keras

# use sub directories names as classes
classes = [i.split(os.path.sep)[1] for i in glob.glob(os.path.join(in_dir,'train/*'))]
classes.sort()

INSHAPE=(NBFRAME,) + SIZE + (CHANNELS,) # (5, 112, 112, 3)
model = action_model(INSHAPE, len(classes))
optimizer = keras.optimizers.Adam(0.001)
model.compile(
    optimizer,
    'categorical_crossentropy',
    metrics=['acc']
)

print(model.summary())

EPOCHS=50

callbacks = [
    keras.callbacks.ReduceLROnPlateau(verbose=1),
    keras.callbacks.ModelCheckpoint(
        '/home/nabaruns/chkp/weights.{epoch:02d}-{val_loss:.2f}.hdf5',
        verbose=1),
]

model.fit_generator(
    train_it,
    validation_data=val_it,
    verbose=1,
    epochs=EPOCHS,
    callbacks=callbacks
)