# -*- coding: utf-8 -*-
"""
Created on Thu Apr 26 21:19:43 2018

@author: jakpo_000
"""


#from copy import deepcopy
#from keras.models import Sequential
from keras.models import load_model
import matplotlib.pyplot as plt
import vis_core as vc
import vis_utils as vu
from debug import DEBUG
    

class Model_explorer():
    """
    layer layer_max, weights get/set, input tensor
    """
    def __init__(self, arg):

        debug = DEBUG()
        debug.pause()

        if isinstance(arg, str):
            self.model = load_model(arg)
            debug.time('Model load from path')
        elif arg.__class__.__name__ is 'Sequential':
            self.model = arg
            debug.time('Sequential model')
        else:
            print('input not supported! Init with Sequential or path to *.h5 Sequential')
            return    
        
    #def _get_weights_bias(model):
    def set_test_image(self, img_path):
        t_size = self.model.input_shape[2], self.model.input_shape[-2]
        self.img = vu.load_2d(img_path, t_size)
        self.tensor = vu.to_tensor(self.img)
        
    def filters(self):
        vc.filters(self.model)
        
    def activations(self):
        a = vc.activations(self.model, self.tensor)
        vu.plot_tensor(a[2])
        
        
    def cam(self):
        vc.cam(self.model, self.tensor)
        
    def grad_ascent(self):
        #ga = vc.gradient_ascent(self.model) 
        stack = []
        for i in range(72):
            #stack.append(n_max(model, filter_index=i))
            stack.append(vc.gradient_ascent(self.model,filter_index=i))
       
        vu.plot_stack(stack)
    
  