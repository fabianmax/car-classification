import re
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


def open_file_structure(path, clean=True):
    """
    Open and prepare file structure

    Args:
        path: path to file containing picture naming schema
        clean: If filename should be cleaned (e.g. to lower cases)

    Returns:
         List cof strings giving the header names
    """

    with open(path) as f:
        structure = f.read()

    structure = [x.replace("'", "") for x in structure.split("', ")]

    if clean:
        # To lower, remove special characters and whitespaces
        structure = [x.lower() for x in structure]
        structure = [re.sub('[^a-zA-Z0-9 \n\.]', '', x) for x in structure]
        structure = [x.replace(' ', '_') for x in structure]
        structure = [x.replace('__', '_') for x in structure]

    return structure


def expand_column(df, column, col_names):
    """
    Extract columns for single string

    Args:
        df: Data.Frame with raw information
        column: Name of column to be expanded into several columns
        col_names: List of names used as new column names

    Returns:
        A pandas DataFrame object
    """
    df_expanded = pd.DataFrame(df[column].str.split('_', expand=True))
    df_expanded.columns = [*col_names, 'id']
    return df_expanded


def show_image(image, label=None, shape=False):
    """
    Function to show image

    Args:
        image:
        label:
        shape:

    Returns:
        None
    """
    plt.figure()
    plt.imshow(image)
    plt.axis('off')

    if label is not None:
        if shape:
            plt.title(label.numpy().decode('utf-8') + ' ' + image.numpy().shape)
        else:
            plt.title(label.numpy().decode('utf-8'))
    elif shape:
        plt.title(image.numpy().shape)


def show_batch(ds, batch_size=32, scaled=True):
    """
    Function to show a batch of images including labels from tf.data object
    """

    plt.figure(figsize=(10, 10))

    for image, label in ds.take(1):
        image_array = image.numpy()
        label_array = label.numpy()
        for idx in range(batch_size):
            ax = plt.subplot(np.ceil(batch_size / 4), 4, idx + 1)
            if scaled:
                plt.imshow(image_array[idx] * 255)
            else:
                plt.imshow(image_array[idx])
            plt.title(label_array[idx].decode('UTF-8') + ' ' + str(image_array[idx].shape), fontsize=10)
            plt.axis('off')

