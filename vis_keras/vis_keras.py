# -*- coding: utf-8 -*-
"""
Created on Thu Apr 26 21:19:43 2018
Contains Model_explorer class. This class provides an easy investigation tool
for a keras sequential models.
@author: jakpo_000
"""


from copy import deepcopy
# from keras.models import Sequential
from keras.models import load_model
import matplotlib.pyplot as plt
import numpy as np
from . import vis_core as vc
from . import vis_utils as vu
# from debug import DEBUG
import os



class Model_explorer():
    """
    With this class a keras sequential 2D/3D grayscale model can be
    investigated. The class supports different visualization techniques and
    methods. From classical methods like filter kernel, activation maps to
    gradient ascent and grad_cam.

    """
    def __init__(self, arg):
        """
        Init with sequential 2D/3D grayscale model or path to .h5 file

        """
        # debug = DEBUG()
        # debug.pause()

        if isinstance(arg, str):
            self.model = load_model(arg)
            self.path_name = os.path.basename(arg)
            # debug.time('Model load from path')
        elif arg.__class__.__name__ is 'Sequential':
            self.model = arg
            self.path_name = 'not from path'
            # debug.time('Sequential model')
        else:
            print('input not supported! Init with Sequential or path to *.h5 Sequential')
            return

        # Mirror model attributes
        self.name = self.model.name
        self.input_shape = self.model.input_shape
        self.input_image_dim = vu.model_helper.model_indim(self.model)
        self.input_image_str = vu.model_helper.model_input(self.model)
        if self.input_image_dim is 2:
            self.t_size = self.input_shape[1], self.input_shape[2]
        elif self.input_image_dim is 3:
            self.t_size = self.input_shape[1], self.input_shape[2], self.input_shape[3]

        self.active_object = False
        self.object = None
        self.object_name = None
        self.summary = lambda: self.model.summary()
        Model_explorer.info(self)

    # def _get_weights_bias(model):
    def info(self):
        print('Name: %s' % self.name)
        print('Path_name: %s' % self.path_name)
        print('Input is %s with shape %s' % (self.input_image_str, self.t_size))


    def set_image_from_path(self, img_path, name=None):
        self.object_name = name
        if self.object_name is None:
            self.object_name = 'unnamed'

        self.object, self.path_str = vu.io.ext_load(img_path, Model_explorer.t_size)
        self.active_object = True

    def set_image_from_array(self, array, name=None):
        ## check size
        self.object_name = name
        if self.object_name is None:
            self.object_name = 'unnamed'

        self.object = array
        self.active_object = True

    def filters(self):
        """
        shows the first conv layer kernels
        """
        if self.active_object is False:
            print('Error! No test object found, set first')
            return

        weights = vc.filters(self.model)
        return weights
        #vu.plot.plot_tensor(weights, weights=True, cmap='gray')


    def activations(self, plot=True, layer=0):
        if self.active_object is False:
            print('Error! No test object found, set first')
            return

        else:
            activation = vc.activations(self.model, self.object)
            if plot:
                if self.input_image_dim is 3:
                    vu.plot.plot_5dtensor(activation[layer])
                elif self.input_image_dim is 2:
                    vu.plot.plot_tensor(activation[layer])

            return activation

    def occ_info(self):
        vu.model_helper.possible_kernel_stride(self.t_size, plot=True)

    def occlusion(self, kernel=None, stride=None):
        if (kernel or stride) is None:
            combinations = vu.model_helper.possible_kernel_stride(self.t_size)
            le = len(combinations)
            le = int(le/2)
            kernel = combinations[le][1]
            stride = combinations[le][2]
            print('Kernel %i and stride %i were choosed automatically!')
        heatmap = vc.occlusion(self.model, self.object, stride, kernel)
        return heatmap

    def grad_cam(self, save_imposed=False, plot_first=True):

        if self.active_object is None:
            print('Error! No test object found, set first')
            return

        heatmap = vc.grad_cam(self.model, self.object)

        if save_imposed:
                base = os.path.basename(self.path_str)
                base = os.path.splitext(base)[0]
                name_str = 'Heatmap-'+ self.object_name
                vu.plot.superimpose(heatmap, self.path_str, save=True, name=name_str)
                print(1)
        return heatmap

    def grad_ascent(self, filter_index=0 ):
        # ga = vc.gradient_ascent(self.model)

            # stack.append(n_max(model, filter_index=i))
        maximized=vc.gradient_ascent(self.model, filter_index, layer_name=None)
        #vu.plot.plot_stack(stack)
        return maximized

    def predict(self):
        pred = self.model.predict(self.object)
        return pred




