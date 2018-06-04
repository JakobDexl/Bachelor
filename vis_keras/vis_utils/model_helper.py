# -*- coding: utf-8 -*-
"""
Created on Wed May 30 15:20:30 2018
Utilities for keras Sequential models
@author: jakpo_000
"""
def test():
    print('test')
    
    
def model_indim(model):
    shape = model.input_shape
    shape_l = len(shape)
    if shape_l is 3:
        dim = 1
    if shape_l is 4:
        dim = 2
    if shape_l is 5:
        dim = 3
    return dim

def model_input(model):
    shape = model.input_shape
    shape_l = len(shape)
    string = ''
    if shape_l is 3:
        string = '1d_no_image'
    if shape_l is 4:
        string = '2d'
    if shape_l is 5:
        string = '3d'
    if shape_l is 6:
        string = '4d_vid'
    if shape[-1] is 1:
        string += '_grayscale'
    if shape[-1] is 3:
        string += '_rgb'
    return string

def count_same(model, layer_name='conv'):
    
    conv_classes = {
    'Conv1D',
    'Conv2D',
    'Conv3D',
    'Conv2DTranspose',
    }
    
    core_classes = {
    'Dense',
    'Flatten',
    }
    
    if layer_name is 'conv':
        classes = conv_classes
    elif layer_name is 'core':
        classes = core_classes
    elif isinstance(layer_name, set):
        print('test')
        classes = layer_name
  
    name = []
    layer_type = []
    pos = [] 
    pos_count = 0
    count = 0
    for layer in model.layers:
        if layer.__class__.__name__ in classes:
            count += 1
            pos.append(pos_count)
            name.append(layer.name)
            layer_type.append(layer.__class__.__name__)
        pos_count += 1
    
    return count, pos, name, layer_type        