#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Apr 15 15:00:51 2018
Rudimentary, not optimized keras CNN Model for classifying MRI Brain images into
normal and abnormal
@author: jakob
"""

from keras.models import Sequential
from keras.layers import Conv2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense, Dropout, BatchNormalization
from keras.preprocessing.image import ImageDataGenerator
import keras.backend.tensorflow_backend as K
import model_utils as mu
import vis_keras.vis_utils.generator as io
#import man_k as mank

def main():
    """
    main function
    How to use: specify test, train path and your processing devive
    Run this file or change the specs
    """
    with K.tf.device('/gpu:0'):

        shape = 128#126#128
        kernel = 9#9
        filters = 18
        batch = 16#64
        epoch = 2
        train_path = '/home/jakob/bachelor/Data/2D/PhantomData/TrainSet' # '/home/jakob/bachelor/Data/2D/OASIS/train'
        test_path =  '/home/jakob/bachelor/Data/2D/PhantomData/TestSet' # '/home/jakob/bachelor/Data/2D/OASIS/test'

        model = Sequential()

        model.add(Conv2D(2*filters, (3,3), input_shape=(shape, shape, 1),
                         padding='same', activation='relu'))

        model.add(BatchNormalization())
        model.add(Dropout(0.02))
        model.add(Conv2D(2*filters, (6,6), padding='same', activation='relu'))
        model.add(Conv2D(filters, (kernel, kernel), padding='same', activation='relu'))
        #model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(BatchNormalization())
        model.add(Dropout(0.02))
        model.add(Conv2D(filters, (kernel, kernel), padding='same', activation='relu'))
        model.add(BatchNormalization())
        model.add(Conv2D(int(0.5*filters), (kernel, kernel), padding='same', activation='relu'))
        #model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(BatchNormalization())
        model.add(Dropout(0.02))
        model.add(Conv2D(int(0.5*filters), (kernel, kernel), padding='same', activation='relu'))
        model.add(Conv2D(int(0.5*filters), (kernel, kernel), padding='same', activation='relu'))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(BatchNormalization())
        model.add(Dropout(0.02))

        model.add(Conv2D(int(0.5*filters), (kernel, kernel), padding='same', activation='relu'))
        model.add(Conv2D(int(0.5*filters), (kernel, kernel), padding='same', activation='relu'))
        #model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(BatchNormalization())
        model.add(Dropout(0.02))
        model.add(Conv2D(int(0.5*filters), (kernel, kernel), padding='same', activation='relu'))
        model.add(BatchNormalization())
        model.add(Conv2D(int(5), (kernel, kernel), padding='same', activation='relu'))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(BatchNormalization())
        model.add(Dropout(0.02))
        model.add(Conv2D(int(4), (kernel, kernel), padding='same', activation='relu'))
        model.add(Conv2D(int(3), (kernel, kernel), padding='same', activation='relu'))
        #model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(BatchNormalization())
        model.add(Dropout(0.02))
        model.add(Flatten())

        model.add(Dense(units=128, activation='relu'))
        model.add(Dense(units=64, activation='relu'))
        model.add(Dense(units=128, activation='relu'))
        model.add(Dense(units=1, activation='sigmoid'))

        model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
##
#        train_datagen = ImageDataGenerator(rescale=1./255)
#    #                                       ,shear_range = 0.2,
#    #                                       zoom_range = 0.2,
#    #                                       horizontal_flip = True)
#
#        test_datagen = ImageDataGenerator(rescale=1./255)
#
#
#        training_set = train_datagen.flow_from_directory(train_path,
#                                                         target_size=(shape, shape),
#                                                         color_mode='grayscale',
#                                                         batch_size=batch,
#                                                         class_mode='binary')
#
#        test_set = test_datagen.flow_from_directory(test_path,
#                                                    target_size=(shape, shape),
#                                                    color_mode='grayscale',
#                                                    batch_size=batch,
#                                                    class_mode='binary')
        training_set = io.generator(train_path, batch_size=batch, target_size=(shape,shape))

        test_set = io.generator(test_path, batch_size=batch, target_size=(shape,shape))
#        tr_n =7800
#        ts_n = 2000
        tr_n = training_set.len()
        ts_n = test_set.len()
#        tb_callback = mu.CustomCallback.tensorCall()
#        acti_callback = mu.CustomCallback.activationHistory(model.input_shape[-2])
#        #cb = [tb_callback,acti_callback]
        cb = []
        history = model.fit_generator(training_set,
                                      steps_per_epoch=tr_n,
                                      epochs=epoch,
                                      validation_data=test_set,
                                      validation_steps=ts_n,
                                      callbacks=cb)

        mu.plot_history(history)

        #stack = acti_callback.get_stack()
        #mu.make_gif(stack, layer_to_vis=1)
        return model, training_set

model, train = main()
