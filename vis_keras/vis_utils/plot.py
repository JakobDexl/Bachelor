# -*- coding: utf-8 -*-
"""
Created on Wed May 30 15:31:53 2018

@author: jakpo_000
"""
from copy import deepcopy
from math import sqrt
import cv2

import numpy as np
import matplotlib.pylab as plt


def heatmap(stack, model, plot=True):
    pic_list = deepcopy(stack)
    length = len(pic_list)
    size = int(sqrt(length))

#    if (size % 2) is not :
#        return
    heatvec = np.zeros(length)

    for i in range(length):
        pic = deepcopy(pic_list[i])
        pic = np.expand_dims(pic, axis=0)
        pic = np.expand_dims(pic, axis=4)
        x = model.predict(pic)
        heatvec[i] = x[0, 0]

    heatmap = np.reshape(heatvec, (size, size))

    if plot is True:
        plt.imshow(heatmap)
        plt.colorbar()
        plt.show
    return heatmap


def plot_stack(liste, cmap='viridis'):
    length = len(liste)
    x = 5
    y = (length//x)+1

    plt.figure(figsize=(11, y+(y)))
    for c in range(length):
        plt.subplot(y, x, c+1)
        plt.imshow(liste[c], cmap)
        plt.title('%s von %s' % (c+1, length))
        plt.xticks([])
        plt.yticks([])

    plt.subplots_adjust(bottom=0.1, right=0.8, top=0.9)
    cax = plt.axes([0.85, 0.1, 0.075, 0.8])
    plt.colorbar(cax=cax)
    plt.show()


def plot_tensor(tensor, weights=False, cmap='gray'):
    length = tensor.shape[-1]

    if weights:
        tens = tensor[:, :, 0, :]
    else:
        tens = tensor[0, :, :, :]

    x = 5
    y = (length//x)+1

    plt.figure(figsize=(11, y+(y)))
    for c in range(length):
        plt.subplot(y, x, c+1)
        plt.imshow(tens[:, :, c], cmap)
        plt.title('%s von %s' % (c+1, length))
        plt.xticks([])
        plt.yticks([])

    plt.subplots_adjust(bottom=0.1, right=0.8, top=0.9)
    cax = plt.axes([0.85, 0.1, 0.075, 0.8])
    plt.colorbar(cax=cax)
    plt.show()


def superimpose(heatmap, img_path, save=True, name='Heatmap'):
    img = cv2.imread(img_path)
    heatmap = cv2.resize(heatmap, (img.shape[1], img.shape[0]))
    heatmap = np.uint8(255*heatmap)
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    superimposed = heatmap * 0.4 + img
    # if save:
    cv2.imwrite('G:/%s.jpg' % name, superimposed)
    # plt.imshow(superimposed)


'''
3d plotting area
https://terbium.io/2017/12/matplotlib-3d/
https://www.datacamp.com/community/tutorials/matplotlib-3d-volumetric-data
https://matplotlib.org/gallery/images_contours_and_fields/layer_images.html#sphx-glr-gallery-images-contours-and-fields-layer-images-py
https://matplotlib.org/gallery/animation/animation_demo.html#sphx-glr-gallery-animation-animation-demo-py
https://matplotlib.org/gallery/event_handling/image_slices_viewer.html#sphx-glr-gallery-event-handling-image-slices-viewer-py
https://matplotlib.org/gallery/specialty_plots/mri_demo.html#sphx-glr-gallery-specialty-plots-mri-demo-py
https://matplotlib.org/gallery/mplot3d/voxels_numpy_logo.html#sphx-glr-gallery-mplot3d-voxels-numpy-logo-py
https://matplotlib.org/gallery/mplot3d/bars3d.html#sphx-glr-gallery-mplot3d-bars3d-py
'''


def normalize(arr):
    arr_min = np.min(arr)
    return (arr-arr_min)/(np.max(arr)-arr_min)


def show_histogram(values):
    n, bins, patches = plt.hist(values.reshape(-1), 50, normed=1)
    bin_centers = 0.5 * (bins[:-1] + bins[1:])

    for c, p in zip(normalize(bin_centers), patches):
        plt.setp(p, 'facecolor', plt.cm.viridis(c))

    plt.show()

    # show_histogram(arr)


class IndexTracker(object):
    def __init__(self, ax, X):
        self.ax = ax
        ax.set_title('use scroll wheel to navigate images')

        self.X = X
        rows, cols, self.slices = X.shape
        self.ind = self.slices//2

        self.im = ax.imshow(self.X[:, :, self.ind])
        self.update()

    def onscroll(self, event):
        print("%s %s" % (event.button, event.step))
        if event.button == 'up':
            self.ind = (self.ind + 1) % self.slices
        else:
            self.ind = (self.ind - 1) % self.slices
        self.update()

    def update(self):
        self.im.set_data(self.X[:, :, self.ind])
        self.ax.set_ylabel('slice %s' % self.ind)
        self.im.axes.figure.canvas.draw()


def scroller():
    fig, ax = plt.subplots(1, 1)

    X = np.random.rand(20, 20, 40)
    tracker = IndexTracker(ax, X)
    fig.canvas.mpl_connect('scroll_event', tracker.onscroll)
    plt.show()


def frame(data):
    fig, ax = plt.subplots()

    for i in range(len(data)):
        ax.cla()
        ax.imshow(data[i])
        ax.set_title("frame {}".format(i))
        # Note that using time.sleep does *not* work here!
        plt.pause(0.1)

