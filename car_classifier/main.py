
import numpy as np
import pandas as pd

from car_classifier.preparation.pipeline import construct_dataset
from car_classifier.preparation.utils import show_batch

from tensorflow.keras.applications import ResNet50V2
from tensorflow.keras import Model
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.optimizers import Adam

INPUT_DATA_DIR = 'data/raw/'
INPUT_SHAPE = (212, 320, 3)
BATCH_SIZE = 32

data = construct_dataset(input_path=INPUT_DATA_DIR, batch_size=BATCH_SIZE)

show_batch(data)

data = construct_dataset(input_path=INPUT_DATA_DIR, batch_size=BATCH_SIZE, one_hot_encoding=True)




base_model = ResNet50V2(include_top=False,
                        input_shape=INPUT_SHAPE,
                        weights='imagenet')

base_model.trainable = False

x = base_model.output
x = GlobalAveragePooling2D(data_format='channels_last')(x)
output = Dense(42, activation='softmax')(x)

final_model = Model(base_model.input, output)

final_model.compile(loss="categorical_crossentropy",
                    optimizer=Adam(0.0001),
                    metrics=["accuracy"])

test_batch = data.take(1)

p = final_model.predict(test_batch)

pred = [np.argmax(x) for x in p]

for img, lab in test_batch.as_numpy_iterator():
    actual = np.argmax(lab, axis=1)

pd.DataFrame({'actual': actual, 'pred': pred})