# -*- coding: utf-8 -*-
"""
Created on Tue Apr 24 22:08:11 2018

@author: jakpo_000
"""

from copy import deepcopy
import imageio

from math import sqrt
import cv2
import numpy as np

from keras.preprocessing import image
from keras import backend as K
import matplotlib.pylab as plt

def load_2d(img_path, t_size):
    if len(t_size) is 1:
        t_size = t_size, t_size
    
    img = image.load_img(img_path, target_size=(t_size))
    img = image.img_to_array(img)
    img /= 255.
    img = img[:, :, 0]
    return img

def to_tensor(img):
    tens = deepcopy(img)
    tens = np.expand_dims(tens, axis=0)
    tens = np.expand_dims(tens, axis=4)
    return tens

def image_stack(img_path, size=126, stride=5, kernel=5, plot=False):
    #check = ((size-kernel)/stride)+1 == even
    origin_img = load_2d(img_path, size)
    img_h = origin_img.shape[0]
    img_w = origin_img.shape[1]

    if plot is True:
        plt.figure()
        plt.imshow(origin_img)
        plt.title('original (%s,%s)' % (img_h, img_w))
        plt.xticks([])
        plt.yticks([])
        plt.colorbar()
        plt.show()

    stack = []
    writer = imageio.get_writer('stack.gif', mode='I', loop=20)
    
    for x in range(0, img_h, stride):

        for y in range(0, img_w, stride):
            manipul_img = deepcopy(origin_img)
            manipul_img[x:(x+kernel), y:(y+kernel)] = 0
            stack.append(manipul_img)
            writer.append_data(manipul_img)
    writer.close()
    return stack

def superimpose(heatmap, img_path, save=True, name='Heatmap'):    
    img = cv2.imread(img_path)
    heatmap = cv2.resize(heatmap, (img.shape[1], img.shape[0]))
    heatmap = np.uint8(255*heatmap)
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    superimposed = heatmap * 0.4 + img
    if save:
        cv2.imwrite('%s.jpg' %name, superimposed)
    plt.imshow(superimposed)
    return 

def count_same(model, layer_name):
    
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

def plot_tensor(tensor, weights=False, cmap='viridis'):
    length = tensor.shape[-1]
    
    if weights:
        tens = tensor[:,:,0,:]
    else:
        tens = tensor[0,:,:,:]
        
    x = 5
    y = (length//x)+1

    plt.figure(figsize=(11, y+(y)))
    for c in range(length):
        plt.subplot(y, x, c+1)
        plt.imshow(tens[:,:,c], cmap)
        plt.title('%s von %s' % (c+1, length))
        plt.xticks([])
        plt.yticks([])
   
    plt.subplots_adjust(bottom=0.1, right=0.8, top=0.9)
    cax = plt.axes([0.85, 0.1, 0.075, 0.8])
    plt.colorbar(cax=cax)
    plt.show()
    


# util function to convert a tensor into a valid image
def deprocess_image(x):
    # normalize tensor: center on 0., ensure std is 0.1
    x -= x.mean()
    x /= (x.std() + 1e-5)
    x *= 0.1

    # clip to [0, 1]
    x += 0.5
    x = np.clip(x, 0, 1)

    # convert to RGB array
    x *= 255
    if K.image_data_format() == 'channels_first':
        x = x.transpose((1, 2, 0))
    x = np.clip(x, 0, 255).astype('uint8')
    return x

def normalize(x):
    # utility function to normalize a tensor by its L2 norm
    return x / (K.sqrt(K.mean(K.square(x))) + K.epsilon())

def n_max(model, img=None, filter_index=0, layer_name='conv2d_1'):     

    #layer_name = 'conv2d_18'
    #filter_index = 0 
    
    img_height = 128
    img_width = 128
    channel = 1
    
    iterations = 40
    step_size = 1
    
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
    grads = normalize(grads)
    
    # this function returns the loss and grads given the input picture
    iterate = K.function([input_img], [loss, grads])

    # we start from a gray image with some noise
    if img is not None:
        input_img_data = img
    else: 
        if K.image_data_format() == 'channels_first':
            input_img_data = np.random.random((1, channel, img_width, img_height))
        else:
            input_img_data = np.random.random((1, img_width, img_height, channel))
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
    img = deprocess_image(img)
    return img[:,:,0]

def multi_act(model):
#    normal=load_2d('OASIS/Test/predict/normal.png', 128)
#    normal=to_tensor(normal)
    
    img = image.load_img('OASIS/Test/predict/normal.png', target_size=(128,128))
    img = image.img_to_array(img)
    img /= 255.
    normal = np.expand_dims(img, axis=0)
    print(normal.shape)
    
    stack = []
    for i in range(128):
        #stack.append(n_max(model, filter_index=i))
        stack.append(n_max(model, img=normal ,filter_index=i, layer_name = 'conv2d_3'))
    return stack    
    