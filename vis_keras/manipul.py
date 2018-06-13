# -*- coding: utf-8 -*-
"""
Created on Sun Jun 10 11:49:08 2018

@author: jakpo_000
"""


def kernel(model, new_weights, layer, filter_n):

    true_weights = model.layers[layer].get_weights()[0]
    # true_weights[:,:,:,]
    model.layers[layer].set_weights(weights)
    pass

def bias(model, layer, bias):
    true_bias = model.get_weights()[1]

    pass

def retrain():
    pass

def freeze():
    pass

