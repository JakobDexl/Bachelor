# -*- coding: utf-8 -*-
"""
Created on Wed May 30 15:29:17 2018

@author: jakpo_000
"""
from copy import deepcopy
import os
import numpy as np
from scipy.ndimage import zoom
from time import time

try:
    import cv2
except ImportError:
    print('opencv not found! Functions cv_load, superimpose are not available')
    cv2 = None

try:
    import nibabel as nib
except ImportError:
    print('nibabel not found! Functions nii, nii_batch are not available')
    nib = None

try:
    import SimpleITK as sitk
except ImportError:
    print('SimpleITK not found! Functions mha are not available')
    sitk = None

from keras.preprocessing import image

_FORMATS2D = ['.png', '.jpg', '.jpeg', '.bmp', '.ppm']
_FORMATS3D = ['.nii', '.mha']

def cv_load_2d(img_path, target_size=None):
    '''
    Load an 2D image to an array with the opencv lib.
    # Arguments
        img_path:    Path with filename as string 
        target_size: Output target size
        resize
    # Returns
        img:         (Resized) grayscale image
    '''
    if cv2 is None:
        print('Not available! Install opencv first')
        return
    img_path = convertPath(img_path)
    img = cv2.imread(img_path)
    if target_size is not None:
        img = cv2.resize(img, (target_size))
    img /= 255.
    img = img[:, :, 0]
    return img


def load_2d(img_path, t_size=None):
    '''
    Load an 2D image to an array with the keras image lib.
    # Arguments
        img_path:    Path with filename as string 
        target_size: Output target size
        resize:      Define resize method
    # Returns
        img:         (Resized) grayscale image
    '''
    img_path = convertPath(img_path)
    if not isinstance(t_size, tuple) and t_size is not None:
        t_size = t_size, t_size
    img = image.load_img(img_path, target_size=t_size)
    img = image.img_to_array(img)
    img /= 255.
    img = img[:, :, 0]
    return img


def mha(img_path, target_size=None):
    '''
    Load an file with .mha extension to an 3D array.
    # Arguments
        img_path:    Path with filename as string 
        target_size: Output target size
    # Returns
        out:         (Resized) Volume
    '''
    if sitk is None:
        print('Not available! Install SimpleITK first')
        return
    img_path = convertPath(img_path)
    img = sitk.ReadImage(img_path, sitk.sitk.Float32)
    arr = sitk.GetImageFromArray(img)
    if target_size is not None:
        arr = refit(arr, target_size)
    arr /= 255.
    arr = np.expand_dims(arr, axis=0)
    return arr


def nii(img_path, target_size=None):
    '''
    Load an file with .nii extension to an 3D array.
    # Arguments
        img_path:    Path with filename as string 
        target_size: Output target size
    # Returns
        out:         (Resized) Volume
    '''
    start_time = time()
    if nib is None:
        print('Not available! Install nibabel first')
        return
    img_path = convertPath(img_path)
    img = nib.load(img_path)
    tmp = np.array(img.dataobj)
    if target_size is not None:
        arr = refit(tmp, target_size)
    arr /= 255.
    #arr = np.expand_dims(arr, axis=-1)
    arr = np.expand_dims(arr, axis=0)
    print('nii_time: %f' % (time()-start_time))
    return arr


def pil_load(path, target_size=None):
    print('Currently not supported')
    return


def ext_load(path, target_size=None):
    '''
    Loads an Image based on its extension.
    # Arguments
        path:        Path with filename as string 
        target_size: Output target size, supports single sizes for cubes
    # Returns
        out:         (Resized) Image/Volume
    '''
    path = convertPath(path)
    if check_ext(path, _FORMATS2D):
        if not isinstance(target_size, tuple) and target_size is not None:
            target_size = target_size, target_size

        img = load_2d(path, target_size)
        img = to_tensor(img)
        return img

    if check_ext(path, '.nii'):
        if not isinstance(target_size, tuple) and target_size is not None:
            target_size = target_size, target_size, target_size

        img = nii(path, target_size)
        img = to_tensor(img)
        return img

    elif check_ext(path, '.mha'):
        if not isinstance(target_size, tuple) and target_size is not None:
            target_size = target_size, target_size, target_size

        img = mha(path, target_size)
        img = to_tensor(img)
        return img

    else:
        print('Not supported image format!')
        return


def load(path, t_size=None):

    if t_size is not None:
        if isinstance(t_size, int):
            flag = 'unclear'
        if isinstance(t_size, tuple):
            if len(t_size) == 2:
                flag = '2D'
            if len(t_size) == 3:
                flag = '3D'
            if len(t_size) >= 4:
                print('False target size!')
                return

    path = convertPath(path)

    if os.path.isfile(path):

        img = ext_load(path, t_size)
        path_str = path

    if os.path.isdir(path):
        if t_size is None:
            print('Please enter target size for path allocation')
            return
        n_name = 0
        path_str = []
        path_str_2d = []
        path_str_3d = []

        for root, dirs, files in os.walk(path):
            for file in files:
                if check_ext(file, _FORMATS2D):
                    tmp = os.path.join(root, file)
                    tmp = convertPath(tmp)
                    path_str_2d.append(tmp)

                if check_ext(file, _FORMATS3D):
                    tmp = os.path.join(root, file)
                    tmp = convertPath(tmp)
                    path_str_3d.append(tmp)
        n_name2d = len(path_str_2d)
        n_name3d = len(path_str_3d)

        if (n_name2d + n_name3d) == 0:
            print('No images found!')
            return
        elif flag is 'unclear' and n_name2d == 0:
            flag = '3D'
            t_size = (t_size, t_size, t_size)
        elif flag is 'unclear' and n_name3d == 0:
            flag = '2D'
            t_size = (t_size, t_size)

        if flag is 'unclear':

            print('Found %d 2D and %d 3D images under support!' %
                  (n_name2d, n_name3d))
            print('\nEnter 2 for 2D import/Enter 3 for 3D import'
                  '/(return with exit)')
            while 1:
                var = input()
                if str(var) == '2':
                    n_name = n_name2d
                    path_str = path_str_2d
                    del(path_str_2d)
                    cube = np.zeros((n_name, t_size, t_size), dtype=np.float32)
                    break
                if str(var) == '3':
                    n_name = n_name3d
                    path_str = path_str_3d
                    del(path_str_3d)
                    cube = np.zeros((n_name, t_size, t_size, t_size),
                                    dtype=np.float32)
                    break
                if str(var) == 'exit':
                    return

        elif flag is '2D':
            n_name = n_name2d
            path_str = path_str_2d
            del(path_str_2d)
            cube = np.zeros((n_name, t_size[0], t_size[1]), dtype=np.float32)
        elif flag is '3D':
            n_name = n_name3d
            path_str = path_str_3d
            del(path_str_3d)
            cube = np.zeros((n_name, t_size[0], t_size[1], t_size[2]),
                            dtype=np.float32)

        if n_name > 100:
                print('Warning! Allocation size too big! Use generator'
                      'function or smaller batches')

        # cube = np.zeros((n_name, t_size[0], t_size[1], 0), dtype=np.float32)
        c = 0
        # make/load picture matrix
        for i in path_str:
            img = ext_load(i, t_size)
            if flag is '2D':
                cube[c] = deepcopy(img[0, :, :, 0])
            if flag is '3D':
                cube[c] = deepcopy(img[0, :, :, :, 0])
            c += 1
        img = cube
        img = np.expand_dims(img, axis=-1)

    return img, path_str


def load_generator(path, batch_size=3, target_size=(20,20), shuffle=True,
                   classes=True):
     if os.path.isdir(path):
        if len(target_size) == 2:
            formats = _FORMATS2D
        if len(target_size) == 3:
            formats = _FORMATS3D
#        else:
#            print('Unsupported target_size')
#            return
        n_name = 0
        path_str = []
        dir_str = []
        # classes = []
        path = convertPath(path)

        for root, dirs, files in os.walk(path):
            for file in files:
                if check_ext(file, formats):
                    tmp = os.path.join(root, file)
                    tmp = convertPath(tmp)
                    dir_str.append(dirs)
                    path_str.append(tmp)
        
        return dir_str
        n_name = len(path_str)
        if shuffle:
            index_arr = np.random.permutation(n_name)
        else:
            index_arr = np.arrange(n_name)

        batch = np.zeros((batch_size,)+target_size)
        batch_len = int(len(index_arr)//batch_size)
        if len(index_arr) % batch_size != 0:
            batch_len += 1

        print(batch_len, n_name)

        gen = chunks(index_arr, batch_size)
        for p in gen:

            print(p)
            for i, j in enumerate(p):
                j = int(j)
                img = ext_load(path_str[j], target_size)
                img = np.squeeze(img, axis=-1)
                batch[i] = img

        yield batch


def to_tensor(img):
    tens = np.expand_dims(img, axis=0)
    tens = np.expand_dims(tens, axis=-1)
    return tens


def refit(arr, target_size):
    '''
    Resize an array with scipy zoom function. 2D and 3D capable
    # Arguments
        arr:         Numpy array 
        target_size: Output target size
    # Returns
        out:         Resized image
    '''
    size = np.asarray(arr.shape)
    tsize = np.asarray(target_size)
    if len(size) == len(tsize):
        erg = tsize/size
        out = zoom(arr, erg)
        return out
    else:
        print('DimError! Array and target size dimensions are not equal')

def chunks(L, n):
    '''
    Yield successive n-sized chunks from L
    # Arguments
    # Returns
        generator
    '''
    for i in range(0, len(L), n):
        yield L[i:i + n]


def convertPath(path):
    '''
    Helper function for converting windows seperator 
    # Arguments
        path: Path with filename as string
    # Returns
        converted_path: Converted path string
    '''
    sep = os.path.sep
    if sep != '/':
        converted_path = path.replace(os.path.sep, '/')
    return converted_path


def get_ext(path):
    '''
    Get the extensions delimited with a dot
    # Arguments
        path: Path with filename as string
    # Returns
        ext_list: List with extensions seperated with a prepended point
    '''
    ext_list = []
    a = True
    while a is True:
        path, ext = os.path.splitext(path)
        if ext is not '':
            ext_list.append(ext)
        else:
            a = False

    return ext_list


def check_ext(path, ext):
    '''
    Check if path ends with specific extension string, list or tuple 
    # Arguments
        path: Path with filename as string
        ext:  Extension name as string, list or tuple
    # Returns
        Boolean true or false
    '''
    extensions = get_ext(path)

    if isinstance(ext, (list, tuple)):
        for name in ext:
            if not name.startswith('.'):
                    name = '.' + name
                    print(name)
            for i in extensions:
                if name == i:
                    return True

    elif isinstance(ext, str):
        if not ext.startswith('.'):
            ext = '.' + ext
        for i in extensions:
                if ext == i:
                    return True
    return False
