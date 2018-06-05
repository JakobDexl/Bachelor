# -*- coding: utf-8 -*-
"""
Created on Thu Apr 26 21:19:43 2018
Contains Model_explorer class. This class provides an easy investigation tool
for a keras sequential models.
@author: jakpo_000
"""


# from copy import deepcopy
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
    Init with model or path to .h5 file
    """
    def __init__(self, arg):

       # debug = DEBUG()
       # debug.pause()

        if isinstance(arg, str):
            self.model = load_model(arg)
            # debug.time('Model load from path')
        elif arg.__class__.__name__ is 'Sequential':
            self.model = arg
            # debug.time('Sequential model')
        else:
            print('input not supported! Init with Sequential or path to *.h5 Sequential')
            return

        self.input_shape = self.model.input_shape
        self.summary = lambda: self.model.summary()

    # def _get_weights_bias(model):
    def set_test_object(self, img_path):
        # 2D grayscale

        if len(self.input_shape) is 4:

            t_size = self.input_shape[1], self.input_shape[2]
            self.batch, self.path_str = vu.io.load(img_path, t_size)
        # 3D volume grayscale
        elif len(self.input_shape) is 5:

            t_size = self.input_shape[1], self.input_shape[2], self.input_shape[3]
            self.batch, self.path_str = vu.io.load(img_path, t_size)

       # return self.batch

    def filters(self):
        """
        shows the first conv layer kernels
        """
        weights = vc.filters(self.model)
        vu.plot.plot_tensor(weights, weights=True, cmap='gray')

    def activations(self, plot=True, ):
        a = vc.activations(self.model, self.batch)
        vu.plot.plot_tensor(a[2])

    def grad_cam(self, save_imposed=False, plot_first=True):

        hstack = []
        for i in range(self.batch.shape[0]):
            # tmp = self.batch[i]
            tmp = np.expand_dims(self.batch, axis=0)
            heatmap = vc.grad_cam(self.model, tmp[:, i])
            hstack.append(heatmap)

        if plot_first is True:
            plt.matshow(hstack[0])

        if save_imposed:
            for element, p_str in zip(hstack, self.path_str):

                base = os.path.basename(p_str)
                base = os.path.splitext(base)[0]
                name_str = 'Heatmap-'+ base
                vu.plot.superimpose(element, p_str, save=True, name=name_str)
                print(1)

    def grad_ascent(self):
        # ga = vc.gradient_ascent(self.model)
        stack = []

        for i in range(3):
            # stack.append(n_max(model, filter_index=i))
            stack.append(vc.gradient_ascent(self.model, filter_index=i))

        vu.plot.plot_stack(stack)

    def predict(self):
        pred = self.model.predict(self.batch)
        return pred
