import numpy as np
import os
import pickle
import matplotlib.pyplot as plt

from random import sample
from tf_explain.core.grad_cam import GradCAM
from tf_explain.core.occlusion_sensitivity import OcclusionSensitivity
from car_classifier.modeling import TransferModel
from tensorflow.keras.preprocessing import image

# Global model settings
INPUT_DATA_DIR = 'data/raw_filtered/'
INPUT_SHAPE = (224, 224, 3)
TARGET = 'make'

MODEL_FOLDER = 'models'
MODEL_SAVED_NAME = 'resnet_unfreeze_all_filtered.tf'
MODEL_NAME = 'resnet_unfreeze_all_filtered'
MODEL_VERSION = '1'

# All available training images
files = [file for file in os.listdir(INPUT_DATA_DIR) if file.endswith(".jpg")]
file_paths = [INPUT_DATA_DIR + file for file in files]

# Load list of targets
file = open('models/classes_all_filtered.pkl', 'rb')
classes = pickle.load(file)

# Load model
model = TransferModel('ResNet', (224, 224, 3), classes)
model.load('models/resnet_unfreeze_all_filtered.tf')

# Get random file
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

# Reshape data
img = img.reshape(-1, *img.shape)

# Prepage data
data = (img, None)
label_idx = np.where(np.array(classes) == label)[0][0]

explainer = GradCAM()
grid = explainer.explain(validation_data=data,
                         model=model.model,
                         class_index=label_idx,
                         layer_name='conv5_block3_3_conv')

plt.figure()
plt.imshow(grid)
plt.axis('off')
plt.show()

explainer = OcclusionSensitivity()
grid = explainer.explain(validation_data=data,
                         model=model.model,
                         class_index=label_idx,
                         patch_size=10)

plt.figure()
plt.imshow(grid)
plt.axis('off')
plt.show()
