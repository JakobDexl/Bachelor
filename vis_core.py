#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May 11 12:37:00 2018

@author: jakob
"""
import matplotlib.pyplot as plt
from keras import models
from keras import backend as K
import numpy as np

import os
from copy import deepcopy

import vis_utils as vu

def activations(model, img_tensor):
    
    #getting layer outputs, init new model
    layer_outputs = [layer.output for layer in model.layers[:]]
    activation_model = models.Model(inputs=model.input, outputs=layer_outputs)
    activations = activation_model.predict(img_tensor)

    return activations
      
def cam(model, img_tensor):

    preds = model.predict(img_tensor)
    number = np.argmax(preds[0])
     
    brain_output = model.output[:, number]
    last_conv = lambda x: vu.count_same(x, 'conv') [-2] [-1]
    last_conv_name = last_conv(model)
    last_conv_layer = model.get_layer(last_conv_name)
    
    grads = K.gradients(brain_output, last_conv_layer.output)[0]
    
    pooled_grads = K.mean(grads, axis=(0, 1, 2))
    
    iterate = K.function([model.input],
                         [pooled_grads, last_conv_layer.output[0]])
    
    pooled_grads_value, conv_layer_output_value = iterate([img_tensor])
    features = last_conv_layer.output.shape[-1].value
    for i in range(features):
        conv_layer_output_value[:, :, i] *= pooled_grads_value[i]
    
    heatmap = np.mean(conv_layer_output_value, axis=-1)
    
    
    heatmap = np.maximum(heatmap, 0)
    heatmap /= np.max(heatmap)
    plt.matshow(heatmap)
    return heatmap
 
def occlusion(model, img_tensor, stride=None, kernel=None, plot=True):

    size = img_tensor.shape[1]

#test odd/even !!! -->warning    
    #calculate standard kernel and size if not is set
    if kernel is None:
        kernel = int(size//2)
    if stride is None:
        stride = int(size//2)
    
    #calculate the number of testimages produced
    n = int(((size-kernel)/stride)+2)
    l = int(n*n)
    
    #load test_image, get sizes
    
    img_h = img_tensor.shape[1]
    img_w = img_tensor.shape[2]
    
    #initialize heatmap
    heatvec = np.zeros(l)
    count = 0
    
    #pred_zero = deepcopy(img_tensor[0])
    pred = model.predict(img_tensor)
    standard_prediction = pred[0, 0]
    
    print('Kernel:     %s' % (kernel))
    print('Stride:     %s' % (stride))
    print('Filter:     %s' % (l))
    print('Prediction: %s' % (standard_prediction))
    
    for x in range(0, img_h, stride):
       
        for y in range(0, img_w, stride):
            
            manipul_img = deepcopy(img_tensor[0,:,:,0])
            manipul_img[x:(x+kernel), y:(y+kernel)] = 0
            t = vu.to_tensor(manipul_img)
            pred = model.predict(t)
            heatvec[count] = pred[0, 0]
           
            prog = (count/l)*100
            
            if (count % 1) == 0:
                print('\r[%4d/%i] %.2f %%' % (count, l, prog), end='')
            count += 1
            
    heatmap = np.reshape(heatvec, (n, n))
    title='heatmap'
    #title = os.path.basename('heatmap')
    if plot is True:
        plt.imshow(heatmap)
        plt.colorbar()
        plt.title('%s\nstandard_prediction=%s\nsize=%s kernel=%s stride=%s' 
                  % (title, standard_prediction, size, kernel, stride))
        plt.xticks([])
        plt.yticks([])
        plt.show
    return heatmap

def filters(model):
    first_conv = lambda x: vu.count_same(x, 'conv') [1] [0]
    first_conv_count = first_conv(model)
    weights = model.layers[first_conv_count].get_weights()[0]
    vu.plot_tensor(weights, weights=True)
    return weights