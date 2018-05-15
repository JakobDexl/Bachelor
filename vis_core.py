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

def gradient_ascent(model, img=None, filter_index=0, layer_name=None):
    
    img_h = model.input_shape[1]
    img_w = model.input_shape[2]
    channel = 1
    
    iterations = 100
    step_size = 1
    
    if layer_name is None:
        last_conv = lambda x: vu.count_same(x, 'conv') [-2] [-1]
        layer_name = last_conv(model)
    
    # get the symbolic outputs of each "key" layer (we gave them unique names).
    layer_dict = dict([(layer.name, layer) for layer in model.layers])
    input_img = model.input

    # build a loss function that maximizes the activation
    # of the nth filter of the layer considered
    layer_output = layer_dict[layer_name].output
    if K.image_data_format() == 'channels_first':
        loss = K.mean(layer_output[:, filter_index, :, :])
    else:
        loss = K.mean(layer_output[:, :, :, filter_index])
    # compute the gradient of the input picture wrt this loss
    grads = K.gradients(loss, input_img)[0]

    # normalization trick: we normalize the gradient
    # grads /= (K.sqrt(K.mean(K.square(grads))) + 1e-5)
    grads = vu.normalize(grads)
    
    # this function returns the loss and grads given the input picture
    iterate = K.function([input_img], [loss, grads])

    # we start from a gray image with some noise
    if img is not None:
        input_img_data = img
    else: 
        if K.image_data_format() == 'channels_first':
            input_img_data = np.random.random((1, channel, img_h, img_w))
        else:
            input_img_data = np.random.random((1, img_h, img_w, channel))
        input_img_data = (input_img_data - 0.5) * 20 + 128
        
    # run gradient ascent for 20 steps
    for i in range(iterations):
        loss_value, grads_value = iterate([input_img_data])
        input_img_data += grads_value * step_size
        
        print('\r\rCurrent loss value:%.3f , filter: %d ' % (loss_value, filter_index), end='')
        if loss_value <= 0. and i >2:
            # some filters get stuck to 0, we can skip them
            print('break')
            break

    img = input_img_data[0]
    img = vu.deprocess_image(img)
    return img[:,:,0]

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