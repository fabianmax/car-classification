import os
import pandas as pd

from car_classifier.pipeline import construct_ds
from car_classifier.modeling import TransferModel
from car_classifier.utils import show_batch

from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight

from tensorflow.keras.optimizers import Adam

# Global settings
INPUT_DATA_DIR = 'data/cars_filtered_top300/'
INPUT_SHAPE = (224, 224, 3)
BATCH_SIZE = 32
TARGET = 'model'
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
ds_train = construct_ds(input_files=files_train, batch_size=BATCH_SIZE, classes=classes_lower, input_size=INPUT_SHAPE,
                        label_type=TARGET, shuffle=True, augment=True)
ds_valid = construct_ds(input_files=files_valid, batch_size=BATCH_SIZE, classes=classes_lower, input_size=INPUT_SHAPE,
                        label_type=TARGET, shuffle=False, augment=False)
ds_test = construct_ds(input_files=files_test, batch_size=BATCH_SIZE, classes=classes_lower, input_size=INPUT_SHAPE,
                       label_type=TARGET, shuffle=False, augment=False)

# Show examples from one batch
plot_size = (18, 18)

show_batch(ds_train, classes, size=plot_size, title='Training data')
show_batch(ds_valid, classes, size=plot_size, title='Validation data')
show_batch(ds_test, classes, size=plot_size, title='Testing data')

# Init base model and compile
model = TransferModel(base=BASE,
                      shape=INPUT_SHAPE,
                      classes=classes,
                      unfreeze='all')

model.compile(loss="categorical_crossentropy",
              optimizer=Adam(0.0001),
              metrics=["categorical_accuracy"])

class_weights = compute_class_weight('balanced', classes, pd.Series([file.split('_')[0] + "_" + file.split('_')[1] for file in files]))

# Train model using defined tf.data.Datasets
model.history = model.train(ds_train=ds_train, ds_valid=ds_valid, epochs=10, class_weights=class_weights)

# Plot accuracy on training and validation data sets
model.plot()

# Evaluate performance on testing data
model.evaluate(ds_test=ds_test)

