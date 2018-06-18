#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jun 15 08:52:07 2018

@author: jakob
"""
import os
import threading
import numpy as np
from copy import deepcopy
from abc import abstractmethod
from . import io as vio

class Sequence(object):
    """Base object for fitting to a sequence of data, such as a dataset.

    Every `Sequence` must implements the `__getitem__` and the `__len__` methods.
    If you want to modify your dataset between epochs you may implement `on_epoch_end`.
    The method `__getitem__` should return a complete batch.

    # Notes

    `Sequence` are a safer way to do multiprocessing. This structure guarantees that the network will only train once
     on each sample per epoch which is not the case with generators.

    # Examples

    ```python
        from skimage.io import imread
        from skimage.transform import resize
        import numpy as np

        # Here, `x_set` is list of path to the images
        # and `y_set` are the associated classes.

        class CIFAR10Sequence(Sequence):

            def __init__(self, x_set, y_set, batch_size):
                self.x, self.y = x_set, y_set
                self.batch_size = batch_size

            def __len__(self):
                return np.ceil(len(self.x) / float(self.batch_size))

            def __getitem__(self, idx):
                batch_x = self.x[idx * self.batch_size:(idx + 1) * self.batch_size]
                batch_y = self.y[idx * self.batch_size:(idx + 1) * self.batch_size]

                return np.array([
                    resize(imread(file_name), (200, 200))
                       for file_name in batch_x]), np.array(batch_y)
    ```
    """

    @abstractmethod
    def __getitem__(self, index):
        """Gets batch at position `index`.

        # Arguments
            index: position of the batch in the Sequence.

        # Returns
            A batch
        """
        raise NotImplementedError

    @abstractmethod
    def __len__(self):
        """Number of batch in the Sequence.

        # Returns
            The number of batches in the Sequence.
        """
        raise NotImplementedError

    def on_epoch_end(self):
        """Method called at the end of every epoch.
        """
        pass

    def __iter__(self):
        """Create an infinite generator that iterate over the Sequence."""
        while True:
            for item in (self[i] for i in range(len(self))):
                yield item
    def test(self):
        pass

class Iterator(Sequence):
    """Abstract base class for image data iterators.
    # Arguments
        n: Integer, total number of samples in the dataset to loop over.
        batch_size: Integer, size of a batch.
        shuffle: Boolean, whether to shuffle the data between epochs.
        seed: Random seeding for data shuffling.
    """

    def __init__(self, n, batch_size, shuffle, seed):
        self.n = n
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.batch_index = 0
        self.total_batches_seen = 0
        self.lock = threading.Lock()
        self.index_generator = self._flow_index(n, batch_size, shuffle, seed)
        self.seed = seed
        self.index_array = None

    def reset(self):
        self.batch_index = 0

    def _set_index_array(self):
        self.index_array = np.arange(self.n)
        if self.shuffle:
            self.index_array = np.random.permutation(self.n)

    def on_epoch_end(self):
        self._set_index_array()

    def _flow_index(self, n, batch_size=32, shuffle=False, seed=None):
        # Ensure self.batch_index is 0.
        self.reset()
        while 1:
            if self.seed is not None:
                np.random.seed(seed + self.total_batches_seen)
            if self.batch_index == 0:
                index_array = np.arange(n)
                if shuffle:
                    index_array = np.random.permutation(n)

            current_index = (self.batch_index * batch_size) % n
            if n > current_index + batch_size:
                current_batch_size = batch_size
                self.batch_index += 1
            else:
                current_batch_size = n - current_index
                self.batch_index = 0
            self.total_batches_seen += 1
            yield (index_array[current_index: current_index + current_batch_size],
                   current_index, current_batch_size)

    def __getitem__(self, idx):
        if idx >= len(self):
            raise ValueError('Asked to retrieve element {idx}, '
                             'but the Sequence '
                             'has length {length}'.format(idx=idx,
                                                          length=len(self)))
        if self.seed is not None:
            np.random.seed(self.seed + self.total_batches_seen)
        self.total_batches_seen += 1
        if self.index_array is None:
            self._set_index_array()
        index_array = self.index_array[self.batch_size * idx:
                                       self.batch_size * (idx + 1)]
        return self._get_batch(index_array)

    def __len__(self):
        return (self.n + self.batch_size - 1) // self.batch_size  # round up

    def __iter__(self):
        # Needed if we want to do something like:
        # for x, y in data_gen.flow(...):
        return self

    def __next__(self, *args, **kwargs):
        return self.next(*args, **kwargs)


class generator(Iterator):
    """Iterator capable of reading images from a directory on disk.
    # Arguments
        directory: Path to the directory to read images from.
            Each subdirectory in this directory will be
            considered to contain images from one class,
            or alternatively you could specify class subdirectories
            via the `classes` argument.
        image_data_generator: Instance of `ImageDataGenerator`
            to use for random transformations and normalization.
        target_size: tuple of integers, dimensions to resize input images to.
        color_mode: One of `"rgb"`, `"grayscale"`. Color mode to read images.
        classes: Optional list of strings, names of sudirectories
            containing images from each class (e.g. `["dogs", "cats"]`).
            It will be computed automatically if not set.
        class_mode: Mode for yielding the targets:
            `"binary"`: binary targets (if there are only two classes),
            `"categorical"`: categorical targets,
            `"sparse"`: integer targets,
            `"input"`: targets are images identical to input images (mainly
                used to work with autoencoders),
            `None`: no targets get yielded (only input images are yielded).
        batch_size: Integer, size of a batch.
        shuffle: Boolean, whether to shuffle the data between epochs.
        seed: Random seed for data shuffling.
        data_format: String, one of `channels_first`, `channels_last`.
        save_to_dir: Optional directory where to save the pictures
            being yielded, in a viewable format. This is useful
            for visualizing the random transformations being
            applied, for debugging purposes.
        save_prefix: String prefix to use for saving sample
            images (if `save_to_dir` is set).
        save_format: Format to use for saving sample images
            (if `save_to_dir` is set).
    """

    def __init__(self, directory, batch_size=3, target_size=(20, 20), shuffle=True,
                 classes=True, seed=None, class_mode='binary'):

        self.directory = directory
        #self.image_data_generator = image_data_generator
        self.target_size = tuple(target_size)
        self.class_mode = class_mode

        self.classes = classes
        if class_mode not in {'categorical', 'binary', 'sparse',
                              'input', None}:
            raise ValueError('Invalid class_mode:', class_mode,
                             '; expected one of "categorical", '
                             '"binary", "sparse", "input"'
                             ' or None.')

        # first, count the number of samples and classes
        self.samples = 0

        if os.path.isdir(self.directory):
            if len(self.target_size) is 2:
                self.formats = vio._FORMATS2D
                print('Init 2D generator')
            if len(self.target_size) is 3:
                self.formats = vio._FORMATS3D
                print('Init 3D generator')
    #        else:
    #            print('Unsupported target_size')

            self.path_str = []
            self.folder_list = []
            # dirslist = []
            path = vio.convert_path(self.directory)

            for root, dirs, files in os.walk(path):
                for file in files:
                    if vio.check_ext(file, self.formats):
                        tmp = os.path.join(root, file)
                        tmp = vio.convert_path(tmp)
                        self.path_str.append(tmp)
                        self.folder_list.append(vio.get_folder(tmp))
                # dirslist.append(dirs)

            self.dif_classes, self.classes = vio.to_binary(self.folder_list)
            self.num_classes = len(self.dif_classes)
            self.samples = len(self.path_str)
            print('Found %d images belonging to %d classes.' % (self.samples,
                                                                self.num_classes))

            super(generator, self).__init__(self.samples, batch_size, shuffle, seed)

    def names(self):
        return self.path_str

    def classnames(self):
        print(self.dif_classes)
        return self.dif_classes

    def len(self):
        return self.samples

    def next(self):
        with self.lock:
            index_array, current_index, current_batch_size = next(self.index_generator)
        return self._get_batch(index_array)

    def _get_batch(self, index_array):
        """For python 2.x.
        # Returns
            The next batch.
        """
        # The transformation of images is not under thread lock
        # so it can be done in parallel
        batch_x = np.zeros((len(index_array),) + self.target_size, dtype=np.float32)
        batch_x = np.expand_dims(batch_x, axis=-1)
        # build batch of image data
        for i, j in enumerate(index_array):
            img = vio.ext_load(self.path_str[j], self.target_size)
            batch_x[i] = deepcopy(img)

        self.classes = np.asarray(self.classes, dtype=np.float32)
        # build batch of labels
        if self.class_mode == 'input':
            batch_y = batch_x.copy()
        elif self.class_mode == 'sparse':
            batch_y = self.classes[index_array]
        elif self.class_mode == 'binary':
            batch_y = self.classes[index_array]
        elif self.class_mode == 'categorical':
            batch_y = np.zeros((len(batch_x), self.num_classes),dtype=np.float32)
            for i, label in enumerate(self.classes[index_array]):

                batch_y[i, int(label)] = 1.
        else:
            return batch_x

        return batch_x, batch_y