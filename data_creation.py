from __future__ import division
from glob import glob
import math
import numpy as np
import os
import pickle


def load_icon_data(data_path, pattern='LLD*.pkl', single_file=None):
    """Loads icon data from pickle files in directory specified in data_path
        pattern:        file name pattern to search for ['LLD_favicon_data*.pkl']
        single_file:    If not None, only the file with the specified number (modulo total number of files) 
                        is loaded [None]
        Returns the numpy arrays of shape (num_icons, 32, 32, 3) of dtype uint8

	Example use with PIL:
	icons = load_icon_data(data_path)
	img = PIL.Image.fromarray(icons[0])
	img.show()"""
    files = glob(os.path.join(data_path, pattern))
    files.sort()
    if single_file is None:
        with open(files[0], 'rb') as f:
            icons = pickle.load(f,encoding='latin-1')
        if len(files) > 1:
            for file in files[1:]:
                with open(file, 'rb') as f:
                    icons_loaded = pickle.load(f,encoding='latin-1')
                icons = np.concatenate((icons, icons_loaded))
    else:
        with open(files[single_file % len(files)], 'rb') as f:
            icons = pickle.load(f,encoding='latin-1')
    return icons


# def save_icon_data(icons, data_path, package_size=100000):
#     if not os.path.exists(data_path):
#         os.makedirs(data_path)
#     num_packages = int(math.ceil(len(icons) / package_size))
#     num_len = len(str(num_packages))
#     for p in range(num_packages):
#         with open(os.path.join(data_path, 'LLD-icon_data_' + str(p).zfill(num_len) + '.pkl'), 'wb') as f:
#             pickle.dump(icons[p*package_size:(p+1)*package_size], f, protocol=pickle.HIGHEST_PROTOCOL)
#

dataset=load_icon_data('//Users//saumya//Desktop//hcml_project//Logos27_DCGAN//LLD-icon')
print(dataset.shape)
print(type(dataset))
np.save('icon_dataset.npy',dataset)

