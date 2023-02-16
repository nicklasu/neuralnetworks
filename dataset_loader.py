""""KMNIST Dataset" (created by CODH), adapted from "Kuzushiji Dataset"
(created by NIJL and others), doi:10.20676/00000341
データセットは「Kuzushiji」と呼ぶMNISTらしいひらがなセットです。
    https://github.com/rois-codh/kmnist"""
import numpy as np
import tensorflow as tf
from mix_up import mix_up


def kkanji_loader(batch_size, mix):
    x_train = np.load('datasets/kkanji-train-imgs.npz')['arr_0']
    y_train = np.load('datasets/kkanji-train-labels.npz')['arr_0']
    x_test = np.load('datasets/kkanji-test-imgs.npz')['arr_0']
    y_test = np.load('datasets/kkanji-test-labels.npz')['arr_0']
    return __data_reshape(x_train, y_train, x_test, y_test, 64, batch_size, mix, 3832)


def kuzushiji49_loader(batch_size, mix):
    x_train = np.load('datasets/k49-train-imgs.npz')['arr_0']
    y_train = np.load('datasets/k49-train-labels.npz')['arr_0']
    x_test = np.load('datasets/k49-test-imgs.npz')['arr_0']
    y_test = np.load('datasets/k49-test-labels.npz')['arr_0']
    return __data_reshape(x_train, y_train, x_test, y_test, 28, batch_size, mix, 49)


def kmnist_loader(batch_size, mix):
    x_train = np.load('datasets/kmnist-train-imgs.npz')['arr_0']
    y_train = np.load('datasets/kmnist-train-labels.npz')['arr_0']
    x_test = np.load('datasets/kmnist-test-imgs.npz')['arr_0']
    y_test = np.load('datasets/kmnist-test-labels.npz')['arr_0']
    return __data_reshape(x_train, y_train, x_test, y_test, 28, batch_size, mix, 10)


def __data_reshape(x_train, y_train, x_test, y_test, shape, batch_size, mix, class_amount):
    x_train = x_train.astype("float32") / 255.0
    x_train = np.reshape(x_train, (-1, shape, shape, 1))
    y_train = tf.one_hot(y_train, class_amount)

    x_test = x_test.astype("float32") / 255.0
    x_test = np.reshape(x_test, (-1, shape, shape, 1))
    y_test = tf.one_hot(y_test, class_amount)

    # Put aside a few samples to create our validation set
    val_samples = 10000
    x_val, y_val = x_train[:val_samples], y_train[:val_samples]
    new_x_train, new_y_train = x_train[val_samples:], y_train[val_samples:]

    val_ds = tf.data.Dataset.from_tensor_slices((x_val, y_val)).batch(batch_size)
    test_ds = tf.data.Dataset.from_tensor_slices((x_test, y_test)).batch(batch_size)

    if mix:
        train_ds_one = (
        tf.data.Dataset.from_tensor_slices((new_x_train, new_y_train))
            .shuffle(batch_size * 100)
            .batch(batch_size)
        )
        train_ds_two = (
            tf.data.Dataset.from_tensor_slices((new_x_train, new_y_train))
            .shuffle(batch_size * 100)
            .batch(batch_size)
        )
        # Because we will be mixing up the images and their corresponding labels, we will be
        # combining two shuffled datasets from the same training data.
        train_ds = tf.data.Dataset.zip((train_ds_one, train_ds_two))
        return mix_up(train_ds, val_ds, test_ds), class_amount

    train_ds = tf.data.Dataset.from_tensor_slices(
        (new_x_train, new_y_train)).shuffle(batch_size * 100).batch(batch_size)
    return (train_ds, val_ds, test_ds), class_amount
