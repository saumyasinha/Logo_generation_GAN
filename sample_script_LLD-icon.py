import h5py
import numpy as np

# simple way to load the complete dataset (for a more sophisticated generator example, see LLD-logo script)
# open hdf5 file
hdf5_file = h5py.File('C:\\Users\Shivendra\Desktop\GAN\LLD-icon.hdf5', 'r')
# load data into memory as numpy array
images, labels = (hdf5_file['data'][:], hdf5_file['labels/resnet/rc_32'][:])
np.save('icon_dataset_from_hd5.npy',images)
np.save('rc_cluster_icon_dataset_from_hd5.npy',labels)

# alternatively, h5py objects can be used like numpy arrays without loading the whole dataset into memory:
# images, labels = (hdf5_file['data'], hdf5_file['labels/resnet/rc_64'])
# here, images[0] will be again returned as a numpy array and can eg. be viewed with matplotlib using
# plt.imshow(images[0])