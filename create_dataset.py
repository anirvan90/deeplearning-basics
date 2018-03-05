from random import shuffle
import glob
import numpy as np
import h5py
import cv2

# Shuffle Data
shuffle_data = True

full_source_path = 'data/train/*.jpg'
sample_source_path = 'data/sample/*.jpg'

full_dest_path = 'data/dogsvcats.hdf5'
sample_dest_path = 'data/dogsvcats-sample.hdf5'

# Where the HDF5 file is going - Full
hdf_path = full_dest_path
cat_dog_train_path = full_source_path

# Where the HDF5 file is going - Sample
# hdf_path = sample_dest_path
# cat_dog_train_path = sample_source_path


data_order = 'tf'
# Read Addresses and Labels from the 'train' folder
addrs = glob.glob(cat_dog_train_path)

# Set 0 for Cat and 1 for Dog
labels = [0 if 'cat' in addr else 1 for addr in addrs]

if shuffle_data:
    c = list(zip(addrs, labels))
    shuffle(c)
    addrs, labels = zip(*c)

# Divide the dataset into 60% Train, 20% Valid and 20% Test
train_addrs = addrs[0:int(0.6 * len(addrs))]
train_labels = labels[0:int(0.6 * len(labels))]

val_addrs = addrs[int(0.6 * len(addrs)):int(0.8 * len(addrs))]
val_labels = labels[int(0.6 * len(addrs)):int(0.8 * len(addrs))]

test_addrs = addrs[int(0.8 * len(addrs)):]
test_labels = labels[int(0.8 * len(labels)):]


train_shape = (len(train_addrs), 224.0, 224.0, 3)
val_shape = (len(val_addrs), 224.0, 224.0, 3)
test_shape = (len(test_addrs), 224.0, 224.0, 3)


hdf5_train_file = h5py.File(hdf_path, mode='w')

hdf5_train_file.create_dataset("train_img", train_shape, np.uint8)
hdf5_train_file.create_dataset("val_img", val_shape, np.uint8)
hdf5_train_file.create_dataset("test_img", test_shape, np.uint8)

hdf5_train_file.create_dataset("train_mean", train_shape[1:], np.uint8)

hdf5_train_file.create_dataset(
    "train_labels", (len(train_addrs),), np.uint8)
hdf5_train_file["train_labels"][...] = train_labels

hdf5_train_file.create_dataset("val_labels", (len(val_addrs),), np.uint8)
hdf5_train_file["val_labels"][...] = val_labels

hdf5_train_file.create_dataset("test_labels", (len(test_addrs),), np.uint8)
hdf5_train_file["test_labels"][...] = test_labels


# mean = np.zeros(train_shape[1:], np.uint8)

for i in range(len(train_addrs)):

    if i % 1000 and i > 1:
        print('Train Data: {}/{}'.format(i, len(train_addrs)))

    addr = train_addrs[i]
    img = cv2.imread(addr)
    img = cv2.resize(img, (224, 224), interpolation=cv2.INTER_CUBIC)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    hdf5_train_file["train_img"][i, ...] = img[None]
    #mean += img / float(len(train_labels))

for i in range(len(val_addrs)):

    if i % 1000 and i > 1:
        print('Val Data: {}/{}'.format(i, len(val_addrs)))

    addr = val_addrs[i]
    img = cv2.imread(addr)
    img = cv2.resize(img, (224, 224), interpolation=cv2.INTER_CUBIC)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    hdf5_train_file["val_img"][i, ...] = img[None]

for i in range(len(test_addrs)):
    if i % 1000 and i > 1:
        print('Test Data: {}/{}'.format(i, len(val_addrs)))

    addr = test_addrs[i]
    img = cv2.imread(addr)
    img = cv2.resize(img, (224, 224), interpolation=cv2.INTER_CUBIC)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    hdf5_train_file["test_img"][i, ...] = img[None]
# save the mean and close the hdf5 file

#hdf5_train_file["train_mean"][...] = mean
hdf5_train_file.close()
