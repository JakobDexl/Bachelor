# -*- coding: utf-8 -*-
"""
Created on Wed May 30 15:31:53 2018

@author: jakpo_000
"""
from copy import deepcopy
from math import sqrt
import numpy as np
import matplotlib.pylab as plt
from .import io
from .import preprocess

try:
    import cv2
except ImportError:
    print('opencv not found! Functions cv_load, superimpose are not available')
    cv2 = None

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


def plot_3d(volume, axis=1, cmap='viridis'):

    volume = preprocess.np_clip(volume)
    volume = np.uint8(255*volume)
    shape=volume.shape
    le = shape[axis]
    x = 5
    y = (le//x)+1
    plt.figure(figsize=(11, y+(y)))

    for c in range(le):
        if axis == 0:
            image = volume[c,:,:]

        if axis == 1:
            image = volume[:,c,:]

        if axis == 2:
            image = volume[:,:,c]

        plt.subplot(y, x, c+1)
        plt.imshow(image, cmap, clim=(np.min(volume), np.max(volume)))
        plt.title('%s von %s' % (c+1, le))
        plt.xticks([])
        plt.yticks([])

    plt.subplots_adjust(bottom=0.1, right=0.8, top=0.9)
    cax = plt.axes([0.85, 0.1, 0.075, 0.8])
    plt.colorbar(cax=cax)
    plt.show()


def plot_stack(liste, cmap='viridis'):
    length = len(liste)
    x = 5
    y = (length//x)+1
    plt.figure(figsize=(11, y+(y)))
    liste = normalize(liste)
    for c in range(length):
        plt.subplot(y, x, c+1)
        plt.imshow(liste[c], cmap, clim=(np.min(liste), np.max(liste)))
        plt.title('%s von %s' % (c+1, length))
        plt.xticks([])
        plt.yticks([])

    plt.subplots_adjust(bottom=0.1, right=0.8, top=0.9)
    cax = plt.axes([0.85, 0.1, 0.075, 0.8])
    plt.colorbar(cax=cax)
    plt.show()


def plot_5dtensor(tensor, axis=2, slices=None, cmap='gray'):
    le = tensor.shape[-1]

    x = 5
    y = (le//x)+1
    plt.figure(figsize=(11, y+(y)))

    if slices is None:
        shape = tensor.shape[-2]
        slices = int(shape/2)

    for c in range(le):
        if axis == 0:
            image = tensor[0, slices, :, :, c]

        if axis == 1:
            image = tensor[0, :, slices, :, c]

        if axis == 2:
            image = tensor[0, :, :, slices, c]

        plt.subplot(y, x, c+1)
        plt.imshow(image, cmap)  # , clim=(np.min(), np.max(volume)))
        plt.title('%s von %s' % (c+1, le))
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


def superimpose(heatmap, img_path, dest_path, name='Heatmap', axis=1):
    img = io.ext_load(img_path, target_size=(128,128,128))

    shape = img.shape[1:-1]


    if len(shape) == 2:
        heatmap = cv2.resize(heatmap, shape)
        heatmap = np.uint8(255*heatmap)
        heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
        superimposed = heatmap * 0.4 + img
        cv2.imwrite(dest_path + '%s.jpg' % name, superimposed)

    # plt.imshow(superimposed)
    else:
        heatmap = io.refit(heatmap, shape)

        #heatmap = np.uint8(255*heatmap)
        for i in range(shape[axis]):
            if axis == 0:
                image = heatmap[i,:,:]
                o_image = img[0,i,:,:,:]
            if axis == 1:
                image = heatmap[:,i,:]
                o_image = img[0,:,i,:,:]
            if axis == 2:
                image = heatmap[:,:,i]
                o_image = img[0,:,:,i,:]

            image = preprocess.np_clip(image)
            o_image = preprocess.np_clip(o_image)
            image = np.uint8(255*image)
            o_image = np.uint8(255*o_image)
            image = cv2.applyColorMap(image, cv2.COLORMAP_JET)

            cube = np.dstack((o_image,o_image,o_image))
            superimposed = (image * 0.4) + cube

            cv2.imwrite(dest_path + '%s_%i.jpg' % (name, i), superimposed)



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
    tmp1 = (arr-arr_min)
    tmp2 = (np.max(arr)-arr_min)
    if tmp2 == 0:
        return arr
    else:
        erg = tmp1 / tmp2
        return erg

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

