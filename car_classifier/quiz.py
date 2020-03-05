import inquirer
import os
import pickle
import numpy as np

import tensorflow as tf
import matplotlib.pyplot as plt

from tensorflow.keras.preprocessing import image
from random import sample


# Global settings
INPUT_DATA_DIR = 'data/raw_filtered/'
INPUT_SHAPE = (224, 224, 3)
TARGET = 'make'

# All available training images
files = [file for file in os.listdir(INPUT_DATA_DIR) if file.endswith(".jpg")]
file_paths = [INPUT_DATA_DIR + file for file in files]

# Create a list of all possible outcomes
if TARGET == 'make':
    classes = list(set([file.split('_')[0] for file in files]))
if TARGET == 'model':
    classes = list(set([file.split('_')[0] + '_' + file.split('_')[1] for file in files]))

file = open('models/classes_all_filtered.pkl', 'rb')
classes = pickle.load(file)

# Load trained model
model = tf.keras.models.load_model('models/resnet_unfreeze_all_filtered.tf')


def quiz():
    """
    Interactive challenge of human (manual input) vs. model

    Returns:
        No return
    """

    while True:

        # Get file
        path = sample(file_paths, 1)[0]

        # Get label
        parts = path.split('/')[-1]
        label = parts.split('_')[0]

        # Prepare image
        img = image.load_img(path, target_size=(224, 224))
        img = image.img_to_array(img)
        img /= 255.0

        # Show image
        plt.figure()
        plt.imshow(img)
        plt.axis('off')
        plt.show()

        # Get user input
        questions = [
            inquirer.List('make',
                          message="What car make is it?",
                          choices=classes,
                          ),
        ]
        answer = inquirer.prompt(questions)

        # Model prediction
        img = img.reshape(-1, *img.shape)
        p = model.predict(img)
        y_hat_idx = np.argmax(p)
        y_hat = classes[y_hat_idx]

        # Outputs
        print(f'Your choice: {answer}')
        print(f'Model prediction: {y_hat} (with {round(np.max(p)*100, 1)}%)')
        print(f'Correct label: {label}')


if __name__ == '__main__':
    quiz()
