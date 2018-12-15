from torchvision import datasets, transforms
from torch.utils.data import Dataset
from matplotlib import pyplot as plt
import cv2
import numpy as np
import random
import scipy.ndimage as ndi
from tqdm import tqdm
import os
from PIL import Image


class ScaledMNIST(Dataset):
    def __init__(self, train=True, transform=None, target_transform=None):
        self.transform = transform
        self.target_transform = target_transform
        self.train = train  # training set or test set

        if not os.path.exists('input/scaled_mnist_train.npz'):
            train_dataset = datasets.MNIST('~/data', train=True, download=True)
            train_imgs, train_labels = train_dataset.train_data.numpy(), train_dataset.train_labels.numpy()

            scaled_train_imgs = []
            for i in tqdm(range(len(train_imgs))):
                img = np.pad(train_imgs[i], 14, 'constant')
                img = random_zoom(img[:,:,np.newaxis], (0.5, 1.5))
                scaled_train_imgs.append(img[:,:,0])
            scaled_train_imgs = np.array(scaled_train_imgs)

            np.savez('input/scaled_mnist_train.npz', images=scaled_train_imgs, labels=train_labels)

        if not os.path.exists('input/scaled_mnist_test.npz'):
            test_dataset = datasets.MNIST('~/data', train=False, download=True)
            test_imgs, test_labels = test_dataset.test_data.numpy(), test_dataset.test_labels.numpy()

            scaled_test_imgs = []
            for i in tqdm(range(len(test_imgs))):
                img = np.pad(test_imgs[i], 14, 'constant')
                img = random_zoom(img[:,:,np.newaxis], (0.5, 1.5))
                scaled_test_imgs.append(img[:,:,0])
            scaled_test_imgs = np.array(scaled_test_imgs)

            np.savez('input/scaled_mnist_test.npz', images=scaled_test_imgs, labels=test_labels)

        if self.train:
            scaled_mnist_train = np.load('input/scaled_mnist_train.npz')
            self.train_data = scaled_mnist_train['images']
            self.train_labels = scaled_mnist_train['labels']
        else:
            scaled_mnist_test = np.load('input/scaled_mnist_test.npz')
            self.test_data = scaled_mnist_test['images']
            self.test_labels = scaled_mnist_test['labels']

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        if self.train:
            img, target = self.train_data[index], self.train_labels[index]
        else:
            img, target = self.test_data[index], self.test_labels[index]

        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        img = Image.fromarray(img, mode='L')

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target

    def __len__(self):
        if self.train:
            return len(self.train_data)
        else:
            return len(self.test_data)


def transform_matrix_offset_center(matrix, x, y):
    o_x = float(x) / 2 + 0.5
    o_y = float(y) / 2 + 0.5
    offset_matrix = np.array([[1, 0, o_x], [0, 1, o_y], [0, 0, 1]])
    reset_matrix = np.array([[1, 0, -o_x], [0, 1, -o_y], [0, 0, 1]])
    transform_matrix = np.dot(np.dot(offset_matrix, matrix), reset_matrix)
    return transform_matrix


def apply_transform(x, transform_matrix, channel_axis=0,
                    fill_mode='nearest', cval=0.):
    x = np.rollaxis(x, channel_axis, 0)
    final_affine_matrix = transform_matrix[:2, :2]
    final_offset = transform_matrix[:2, 2]
    channel_images = [ndi.interpolation.affine_transform(
        x_channel,
        final_affine_matrix,
        final_offset,
        order=0,
        mode=fill_mode,
        cval=cval) for x_channel in x]
    x = np.stack(channel_images, axis=0)
    x = np.rollaxis(x, 0, channel_axis + 1)
    return x


def random_zoom(X, zoom_range, row_axis=0, col_axis=1, channel_axis=2,
                fill_mode='nearest', cval=0.):
    if len(zoom_range) != 2:
        raise ValueError('`zoom_range` should be a tuple or list of two floats. '
                         'Received arg: ', zoom_range)

    z = np.random.uniform(zoom_range[0], zoom_range[1])
    zoom_matrix = np.array([[z, 0, 0],
                            [0, z, 0],
                            [0, 0, 1]])

    h, w = X.shape[row_axis], X.shape[col_axis]
    transform_matrix = transform_matrix_offset_center(zoom_matrix, h, w)
    X = apply_transform(X, transform_matrix, channel_axis, fill_mode, cval)

    return X
