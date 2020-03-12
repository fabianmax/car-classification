import tensorflow as tf

from tensorflow.keras.applications.resnet_v2 import preprocess_input


def get_label(filename: str, label_type: str = 'make'):
    """
    Function to get label from filename

    Args:
        filename: file name to extract label from
        label_type: which type of label is needed (either 'make' or 'model')

    Returns:
        tf.tensor with label

    Raises:
        ValueError: If illegal value argument
    """

    # Split full path in parts, make and model are contained in the last part
    parts = tf.strings.split(filename, '/')
    name = parts[-1]
    label = tf.strings.split(name, '_')

    # The make is the first element of the filename
    if label_type == 'make':
        return tf.strings.lower(label[0])
    # Model is the second element of the filename; model and make have to be combined
    elif label_type == 'model':
        return tf.strings.lower(label[0] + '_' + label[1])
    else:
        raise ValueError('label must be either "make" or "model" and not ', label_type)


def get_image(filename: str, size: tuple = (212, 320)):
    """
    Function to load image as tensor and resize it to specific size

    Args:
        filename: file name (path)
        size: tuple (height, width) with size of image after resizing

    Returns:
        Image as tf.Tensor
    """
    image = tf.io.read_file(filename)
    image = tf.image.decode_jpeg(image)
    # Use cast instead of convert_image_dtype to avoid rescaling
    image = tf.cast(image, tf.float32)
    image = tf.image.resize_with_crop_or_pad(image, target_height=size[0], target_width=size[1])
    image = preprocess_input(image)
    return image


def parse_file(filename: str, classes: list, input_size: tuple, label_type: str):
    """
    Function to parse files; loading images from file path and creating (one hot encoded) labels

    Args:
        filename: filename (including full path)
        classes: list of all classes for encoding
        input_size: size of images (output size)
        label_type: 'make' or 'model'

    Returns:
        tuple of image and (one hot encoded) label
    """
    label = get_label(filename, label_type=label_type)
    image = get_image(filename, size=input_size)

    target = one_hot_encode(classes=classes, label=label)

    return image, target


def one_hot_encode(classes: list, label):
    """
    Function for one hot encoding of label, given a full list of possible classes

    Args:
        classes: list of all classes for encoding
        label: label to be encoded

    Returns:
        Encoded label
    """
    n_classes = len(classes)
    names = tf.constant(classes, shape=[1, n_classes])
    index = tf.argmax(tf.cast(tf.equal(names, label), tf.int32), axis=1)
    y = tf.one_hot(index, n_classes)
    y = tf.squeeze(y)
    return y


def image_augment(image, label):
    """
    Function for image augmentation

    Args:
        image: Images from tf.data.Dataset
        label: Labels from tf.data.Dataset

    Returns:
        Tuple of image, label
    """

    image = tf.image.random_flip_left_right(image)
    image = tf.image.random_brightness(image, max_delta=0.1 / 255.0)

    # Make sure the image is still in [0, 1]
    image = tf.clip_by_value(image, 0.0, 1.0)

    return image, label


def construct_ds(input_files: list,
                 batch_size: int,
                 classes: list,
                 label_type: str,
                 input_size: tuple = (212, 320),
                 prefetch_size: int = 10,
                 shuffle_size: int = 32,
                 augment: bool = False):
    """
    Function to construct a tf.data.Dataset set from list of files

    Args:
        input_files: list of files
        batch_size: number of observations in batch
        classes: list with all class labels
        input_size: size of images (output size)
        prefetch_size: buffer size (number of batches to prefetch)
        shuffle_size: shuffle size (size of buffer to shuffle from)
        augment: boolean if image augmentation should be applied
        label_type: 'make' or 'model'

    Returns:
        buffered and prefetched tf.data.Dataset object with (image, label) tuple
    """
    # Create tf.data.Dataset from list of files
    ds = tf.data.Dataset.from_tensor_slices(input_files)

    # Shuffle files
    ds = ds.shuffle(buffer_size=shuffle_size)

    # Load image/labels
    ds = ds.map(lambda x: parse_file(x, classes=classes, input_size=input_size, label_type=label_type))

    # Image augmentation
    if augment:
        ds = ds.map(image_augment)

    # Batch and prefetch data
    ds = ds.batch(batch_size=batch_size)
    ds = ds.prefetch(buffer_size=prefetch_size)

    return ds

