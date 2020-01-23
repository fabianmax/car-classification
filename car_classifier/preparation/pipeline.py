import tensorflow as tf


def get_label(filename, label_type='make'):
    """
    Function to get label from file_name

    Args:
        filename: file name to extract label from
        label_type: which type of label is needed (either 'make' or 'model')

    Returns:
        tf.Tensor with label

    Raises:
        ValueError: If illegal value argument
    """
    parts = tf.strings.split(filename, '/')
    name = parts[-1]
    label = tf.strings.split(name, '_')

    if label_type == 'make':
        return tf.strings.lower(label[0])
    elif label_type == 'model':
        return tf.strings.lower(label[0] + '_' + label[1])
    else:
        raise ValueError('label must be either "make" or "model" and not', label_type)


def get_image(filename, size=(212, 320)):
    """
    Function to lead image as tensor and resize it to specific size

    Args:
        filename: file name (path)
        size: size of image after resizing

    Returns:
        Image as tf.Tensor
    """
    image = tf.io.read_file(filename)
    image = tf.image.decode_jpeg(image)
    image = tf.image.convert_image_dtype(image, tf.float32)
    image = tf.image.resize_with_crop_or_pad(image, target_height=size[0], target_width=size[1])
    image = image / 255.0
    return image


def parse_file(filename):

    return get_image(filename), get_label(filename)


def one_hot_encode(labels, label):
    n_classes = len(labels)
    names = tf.constant(labels, shape=[1, n_classes])
    index = tf.argmax(tf.cast(tf.equal(names, label), tf.int32), axis=1)
    y = tf.one_hot(index, n_classes)
    y = tf.squeeze(y)
    return y


def construct_dataset(input_path, batch_size, prefetch_size=10, one_hot_encoding=False):
    """
    Function to construct a tf.data set from files

    Args:
        input_path: path for training files
        batch_size: number of observations in batch
        prefetch_size: buffer size (number of batches to prefetch)

    Returns:
        buffered and prefetched tf.data object with (image, label)
    """
    # Get list of files
    file_ds = tf.data.Dataset.list_files(input_path + '*.jpg')

    # Parse file and extract image and label from it
    if one_hot_encoding:
        # Get a list of unique classes
        classes = list(set(file_ds.map(get_label).as_numpy_iterator()))
        ds = file_ds.map(lambda x: (get_image(x), one_hot_encode(classes, get_label(x))))
    else:
        ds = file_ds.map(lambda x: (get_image(x), get_label(x)))

    # Parse file and extract image and label from it
    #ds = file_ds.map(parse_file)

    # Shuffle data
    # Create batch of (image, label)
    # and prefetch batches
    ds = ds.shuffle(buffer_size=len(input_path))
    ds = ds.batch(batch_size=batch_size)
    ds = ds.prefetch(buffer_size=prefetch_size)

    return ds





