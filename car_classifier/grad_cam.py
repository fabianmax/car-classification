import os
import pickle
import cv2
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

from random import sample
from car_classifier.modeling import TransferModel
from tensorflow.keras.preprocessing import image

# Global model settings
INPUT_DATA_DIR = 'data/raw_filtered/'
INPUT_SHAPE = (224, 224, 3)

# All available training images
files = [file for file in os.listdir(INPUT_DATA_DIR) if file.endswith(".jpg")]
file_paths = [INPUT_DATA_DIR + file for file in files]

# Load list of targets
file = open('models/classes_all_filtered.pkl', 'rb')
classes = pickle.load(file)

# Load model
model = TransferModel('ResNet', INPUT_SHAPE, classes)
model.load('models/resnet_unfreeze_all_filtered.tf')

# Get random file
sample_path = sample(file_paths, 1)[0]

# Get label
parts = sample_path.split('/')[-1]
label = parts.split('_')[0]

# Load and prepare (normalize) image
img = image.load_img(sample_path, target_size=(224, 224))
img = image.img_to_array(img)
img /= 255.0

# Show image
plt.figure()
plt.imshow(img)
plt.axis('off')
plt.show()

# Reshape image to (batch_size, heigh, width, channel)
img = img.reshape(-1, *img.shape)

# Get label index
label_idx = np.where(np.array(classes) == label)[0][0]

# Gradient model, outputs tuple with:
# - output of conv layer
# - output of head layer
grad_model = tf.keras.models.Model([model.model.inputs],
                                   [model.model.get_layer('conv5_block3_3_conv').output, model.model.output])

# Run model and record outputs, loss, and gradients
with tf.GradientTape() as tape:
    conv_outputs, predictions = grad_model(img)
    loss = predictions[:, label_idx]

# Output of conv layer
output = conv_outputs[0]

# Gradients of loss wrt. conv layer
grads = tape.gradient(loss, conv_outputs)[0]

# Guided Backprop (elimination of negative values)
gate_f = tf.cast(output > 0, 'float32')
gate_r = tf.cast(grads > 0, 'float32')
guided_grads = tf.cast(output > 0, 'float32') * tf.cast(grads > 0, 'float32') * grads

# Average weight of filters
weights = tf.reduce_mean(guided_grads, axis=(0, 1))

# Class activation map (cam)
# Multiply values of conv filters with gradient weights
cam = np.ones(output.shape[0: 2], dtype = np.float32)
for i, w in enumerate(weights):
    cam += w * output[:, :, i]

# Rescale to org image size and min-max scale
cam = cv2.resize(cam.numpy(), (224, 224))
cam = np.maximum(cam, 0)
heatmap = (cam - cam.min()) / (cam.max() - cam.min())

# Original image, reshape back, denormalize and adjust type to heatmap
src_1 = img.reshape(img.shape[1:])
src_1 = np.uint8(255*src_1)
src_1 = cv2.cvtColor(src_1, cv2.COLOR_RGB2BGR)

# Define color map based on heatmap
src_2 = np.uint8(255*heatmap)
src_2 = cv2.applyColorMap(src_2, cv2.COLORMAP_RAINBOW)

# Calculates the weighted sum of two arrays:
# dst = src1*alpha + src2*beta + gamma
output_image = cv2.addWeighted(src1=src_1, alpha=0.5, src2=src_2, beta=1, gamma=0)

# Show
plt.figure()
plt.imshow(output_image)
plt.axis('off')
plt.show()
