import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf


def show_batch(ds: tf.data.Dataset,
               classes: list,
               rescale: bool = False,
               size: tuple = (10, 10),
               title: str = None):
    """
    Function to show a batch of images including labels from tf.data object

    Args:
        ds: a (batched) tf.data.Dataset
        classes: a list of all classes (in order of one-hot-encoding)
        rescale: boolen whether to multiple image values by 255
        size: tuple giving plot size
        title: plot title

    Returns:
        matplotlib.pyplot
    """

    plt.figure(figsize=size)

    # Take on batch from dataset and iterate over image-label-combination
    for image, label in ds.take(1):
        image_array = image.numpy()
        label_array = label.numpy()
        batch_size = image_array.shape[0]
        for idx in range(batch_size):
            label = classes[np.argmax(label_array[idx])]
            ax = plt.subplot(np.ceil(batch_size / 4), 4, idx + 1)
            if rescale:
                plt.imshow(image_array[idx] * 255)
            else:
                plt.imshow(image_array[idx])
            plt.title(label + ' ' + str(image_array[idx].shape), fontsize=10)
            plt.axis('off')

    if title is not None:
        plt.suptitle(title)

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show()


def create_target_list(files: list, target: str = 'make') -> list:
    """
    Create a list of unique target classes from file names

    Args:
        files: a list of file names
        target: either 'model' or 'make'

    Returns:
        list of classes
    """

    if target not in ['make', 'model']:
        raise ValueError('target must be either "make" or "model"')

    if target == 'make':
        classes = list(set([file.split('_')[0] for file in files]))
    if target == 'model':
        classes = list(set([file.split('_')[0] + '_' + file.split('_')[1] for file in files]))

    return classes




