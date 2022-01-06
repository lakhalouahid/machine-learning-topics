import pickle
import cv2 as cv
from utils import *

def load_cifar10_data():
    folder = 'data/cifar-10-batches-py'
    files = [os.path.join(folder, 'data_batch_' + str(i)) for i in range(1, 6)]
    all_images, all_labels = np.zeros((50000, 3072)), np.zeros((50000,))
    for i, file in enumerate(files):
        with open(file, 'rb') as fd:
            data = pickle.load(fd, encoding='bytes')
            all_images[10000*i:10000*(i+1),:] = data[b'data']
            all_labels[10000*i:10000*(i+1)] = np.array(data[b'labels'])

    return np.asarray(all_images, dtype=np.float32), np.asarray(all_labels, dtype=np.float32)

def load_raw_cifar10_data(samples_length, index):
    folder = 'data/cifar-10-batches-py'
    files = [os.path.join(folder, 'data_batch_' + str(i)) for i in range(1, 6)]
    all_images, all_labels = np.zeros((samples_length, 3072)), np.zeros((samples_length,))
    if samples_length <= 10000:
        with open(files[index], 'rb') as fd:
            data = pickle.load(fd, encoding='bytes')
            all_images = data[b'data'][:samples_length]
            all_labels = np.array(data[b'labels'][:samples_length])
    else:
        j = 0
        for i in range(index, index+int(np.ceil(samples_length/10000))):
            with open(files[i], 'rb') as fd:
                data = pickle.load(fd, encoding='bytes')
                all_images[10000*j:10000*(j+1),:] = data[b'data'][:min(10000, samples_length-i*10000)]
                all_labels[10000*j:10000*(j+1)] = np.array(data[b'labels'][:min(10000, samples_length-i*10000)])
                j += 1

    return np.asarray(all_images.swapaxes(0, 1), dtype=np.float32), np.asarray(all_labels, dtype=np.float32)
