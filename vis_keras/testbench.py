# -*- coding: utf-8 -*-
"""
Created on Fri Jun  8 11:45:25 2018
A parent object can explore one model at a time. Sub classes are
investigation objects for example images or batches you would like to test.
These functions are logged in a test configuration. You can have muliple
tests for one model.
@author: jakpo_000
"""

class Model_compare():

    def __init__(self):
        self.model_list = []
        self.test_obj = []
        self.tests = []
        self.active_test = 0
        self.active_object = None
        Model_explorer.add_test(self, 'reference')

        # print('Active test: %s' % (self.tests[self.active_test].name))
        # print('Active test object: %s' % (self.test_obj[self.active_object].name))

    def add_test_object(self, path, name=None):
    if name is None:
        count = len(self.test_obj)
        name = 'Test_object_'+str(count)
    new = Model_explorer.Testobject(name, path)
    self.test_obj.append(new)

    def add_test(self, name):
        new = Model_explorer.Test_config(name)
        self.tests.append(new)

    def set_active_test(self, test):
        self.active_test = test

    def set_active_object(self, obj):
        self.active_object = obj

    class Testobject(object):
        def __init__(self, name, img_path):
            self.name = name
            self.content, self.path_str = vu.io.load(img_path, Model_explorer.t_size)

    class Test_config(object):
        def __init__(self, name):
            self.name = name
