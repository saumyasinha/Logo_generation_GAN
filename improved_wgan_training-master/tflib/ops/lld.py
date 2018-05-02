#!/usr/bin/env python
# -- coding: utf-8 --
import numpy as np

import os
import urllib
import gzip
import _pickle as pickle

# def unpickle(file):
#     with open(file, 'rb') as fo:
#         dict = pickle.load(fo, encoding='bytes')
#     print(dict.keys())
#     return dict[b'data'], dict[b'labels']

def lld_generator(batch_size,images,labels):


    def get_epoch():
        rng_state = np.random.get_state()
        np.random.shuffle(images)
        np.random.set_state(rng_state)
        np.random.shuffle(labels)

        for i in range(int(len(images) / batch_size)):
            yield (images[i*batch_size:(i+1)*batch_size], labels[i*batch_size:(i+1)*batch_size])

    return get_epoch


def load(batch_size,X_train,X_test,y_train,y_test):
    return (
        lld_generator(batch_size,X_train,y_train),
        lld_generator(batch_size,X_test,y_test)
    )