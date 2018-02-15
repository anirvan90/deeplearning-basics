from random import shuffle
import glob

# Shuffle Data
shuffle_data = True

# Where the HDF5 file is going
hdf5_path = 'data/dataset.hdf5'
cat_dog_train_path = 'data/train/*.jpg'

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
val_labels = labels[int(0.6 * len(labels)):int(0.8 * len(addrs))]

test_addrs = addrs[int(0.8 * len(addrs))]
test_labels = labels[int(0.8 * len(labels))]
