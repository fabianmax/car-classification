import os

import numpy as np
import pandas as pd

from car_classifier.pipeline import construct_ds
from car_classifier.modeling import TransferModel
from car_classifier.utils import show_batch

from sklearn.model_selection import train_test_split

from tensorflow.keras.optimizers import Adam

# Gobal settings
INPUT_DATA_DIR = 'data/raw/'
INPUT_SHAPE = (212, 320, 3)
BATCH_SIZE = 32
TARGET = 'make'
BASE = 'ResNet'

# All available training images
files = [file for file in os.listdir(INPUT_DATA_DIR) if file.endswith(".jpg")]
file_paths = [INPUT_DATA_DIR + file for file in files]

# Create a list of all possible outcomes
if TARGET == 'make':
    classes = list(set([file.split('_')[0] for file in files]))
if TARGET == 'model':
    classes = list(set([file.split('_')[0] + '_' + file.split('_')[1] for file in files]))

# Targets in list
classes_lower = [x.lower() for x in classes]

# Split paths into train, valid, and test
files_train, files_test = train_test_split(file_paths, test_size=0.25)
files_train, files_valid = train_test_split(files_train, test_size=0.25)

# Construct tf.data.Dataset from file paths
ds_train = construct_ds(input_files=files_train, batch_size=BATCH_SIZE, classes=classes_lower, input_size=INPUT_SHAPE)
ds_valid = construct_ds(input_files=files_valid, batch_size=BATCH_SIZE, classes=classes_lower, input_size=INPUT_SHAPE)
ds_test = construct_ds(input_files=files_test, batch_size=BATCH_SIZE, classes=classes_lower, input_size=INPUT_SHAPE)

# Show examples from one batch
plot_size = (18, 18)

show_batch(ds_train, classes, size=plot_size, title='Training data')
show_batch(ds_valid, classes, size=plot_size, title='Validation data')
show_batch(ds_test, classes, size=plot_size, title='Testing data')

# Init base model and compile
model = TransferModel(base=BASE, shape=INPUT_SHAPE, classes=classes)
model.compile(loss="categorical_crossentropy",
              optimizer=Adam(0.0001),
              metrics=["categorical_accuracy"])

# Train model using defined tf.data.Datasets
model.train(ds_train=ds_train, ds_valid=ds_valid, epochs=10)

# Plot accuracy on training and validation data sets
model.plot()

# Evaluate performance on testing data
model.evaluate(ds_test=ds_test)


# ---------
test_batch = ds_train.take(1)

p = model.predict(test_batch)

pred = [np.argmax(x) for x in p]

for img, lab in test_batch.as_numpy_iterator():
    actual = np.argmax(lab, axis=1)

pd.DataFrame({'actual': actual, 'pred': pred})
