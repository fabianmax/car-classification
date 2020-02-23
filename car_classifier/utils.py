import re
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from tqdm import tqdm
from google.cloud import storage


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


def show_batch(ds, classes, rescale=True, size=(10, 10), title=None):
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

    plt.tight_layout()
    plt.show()


class GoogleCloudStorage:
    """
    Class for up/down-loading files to Google Cloud Storage
    """

    def __init__(self):
        self.bucket = 'car-classifier'
        self.credentials = 'resources/STATWORX-5db149736e9d.json'
        self.storage_client = storage.Client.from_service_account_json(self.credentials)

    def _upload_blob(self, source_file_name, destination_blob_name):

        bucket = self.storage_client.get_bucket(self.bucket)
        blob = bucket.blob(destination_blob_name)

        blob.upload_from_filename(source_file_name)

    def _download_blob(self, source_blob_name, destination_file_name):

        bucket = self.storage_client.get_bucket(self.bucket)
        blob = bucket.blob(source_blob_name)

        blob.download_to_filename(destination_file_name)

    def upload_files(self, files):

        if isinstance(files, str):

            source = files
            destination = files.split('/')[-1]
            self._upload_blob(source, destination)
            print(f'File {source} uploaded to {destination}')

        elif isinstance(files, list):

            source = files
            destination = [x.split('/')[-1] for x in files]

            for s, d in tqdm(zip(source, destination), total=len(source)):
                self._upload_blob(s, d)

    def download_files(self, source_files, destination_path):

        if isinstance(source_files, str):

            source = source_files
            destination = destination_path + source_files

            self._download_blob(source, destination)
            print(f'File {source} downloaded to {destination}')

        elif isinstance(source_files, list):

            source = source_files
            destination = [destination_path + x for x in source_files]

            for s, d in tqdm(zip(source, destination), total=len(source)):
                self._download_blob(s, d)





