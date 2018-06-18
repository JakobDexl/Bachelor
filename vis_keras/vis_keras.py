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
        self.num_test_obj = 0
        self.active_object = None
        self.generator = None
        self.summary = lambda: self.model.summary()
        Model_explorer.info(self)

    # def _get_weights_bias(model):
    def info(self):
        print('Name: %s' % self.name)
        print('Path_name: %s' % self.path_name)
        print('Input is %s with shape %s' % (self.input_image_str, self.t_size))

    def set_test_object(self, img_path, name=None):
        if name is None:
            count = self.num_test_obj
            name = 'Test_object_'+ str(self.name) + str(count)
        self.active_object, self.path_str = vu.io.load(img_path, Model_explorer.t_size)

    def filters(self):
        """
        shows the first conv layer kernels
        """
        if self.active_object is None:
           print('Error! No test object found, set first')
           return

        weights = vc.filters(self.model)
        #vu.plot.plot_tensor(weights, weights=True, cmap='gray')

    def activations(self, plot=True, ):
        if self.active_object is None:
           print('Error! No test object found, set first')
           return
        return vc.activations(self.model, self.active_object)
        #vu.plot.plot_tensor(a[2])

    def grad_cam(self, save_imposed=False, plot_first=True):

        if self.active_object is None:
           print('Error! No test object found, set first')
           return

        hstack = []

        if self.generator is not None:
            count=0
            tmp=[]
            le=(16*gen.__len__())-9
            for i in range(1):
                a = gen.__getitem__(i)
                for l in a[0]:
                    t=deepcopy(l)
                    t = np.expand_dims(t, axis=0)
                    b = vk.vis_core.grad_cam(model,t,out=arg)
                    h_stack.append(b)
                    print('\r[%i/%i] ' % (count, le), end='')
                    count += 1
        else:
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

        #vu.plot.plot_stack(stack)

    def predict(self):
        pred = self.model.predict(self.batch)
        return pred




