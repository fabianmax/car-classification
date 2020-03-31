import numpy as np
from flask import (
    Flask,  # import main Flask class and request object
    jsonify,
    request)

import cv2
import tensorflow as tf
from modeling import TransferModel
from tensorflow.keras.applications.resnet_v2 import preprocess_input
from tensorflow.keras.preprocessing import image

# Global model settings
INPUT_SHAPE = (224, 224, 3)

# Pseudo classes to allow creation of model instance
classes = [0, 1]

# Load model
model = TransferModel('ResNet', INPUT_SHAPE, classes=classes)
model.load('/models/resnet_unfreeze_all_filtered/1')

# Create instance of flask
app = Flask(__name__)  # create the Flask app


@app.route('/test')
def test():
    return 'Flask server is up and running...'


@app.route('/grad-cam', methods=['POST'])  # GET requests will be blocked
def grad_cam():

    # Read input
    req_data = request.get_json()

    # Label contains filename (not full path)
    label = req_data['label']
    # img (preprocessed?)
    img = req_data['image']

    # Get label index
    label = label.split('_')[0] + '_' + label.split('_')[1]
    label_idx = np.where(np.array(model.classes)[:, 0] == label)[0][0]

    # # Load and prepare (normalize) image
    img = image.img_to_array(img)
    img = tf.image.resize_with_crop_or_pad(img, 224, 224)
    img_to_plot = img / 255.0
    img = preprocess_input(img)
    img = img.numpy()

    # Reshape image to (batch_size, heigh, width, channel)
    img = img.reshape(-1, *img.shape)

    # Gradient model, outputs tuple with:
    # - output of conv layer
    # - output of head layer
    grad_model = tf.keras.models.Model(
        [model.model.inputs],
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
    guided_grads = gate_f * gate_r * grads

    # Average weight of filters
    weights = tf.reduce_mean(guided_grads, axis=(0, 1))

    # Class activation map (cam)
    # Multiply values of conv filters with gradient weights
    cam = tf.reduce_sum(output * weights, axis=2)

    # Rescale to org image size and min-max scale
    cam = cv2.resize(cam.numpy(), (224, 224))
    cam = np.maximum(cam, 0)
    heatmap = (cam - cam.min()) / (cam.max() - cam.min())

    # Original image, reshape back, denormalize and adjust type to heatmap
    src_1 = img_to_plot
    src_1 = np.uint8(255 * src_1)
    src_1 = cv2.cvtColor(src_1, cv2.COLOR_RGB2BGR)

    # Define color map based on heatmap
    src_2 = np.uint8(255 * heatmap)
    src_2 = cv2.applyColorMap(src_2, cv2.COLORMAP_RAINBOW)

    # Calculates the weighted sum of two arrays:
    # dst = src1*alpha + src2*beta + gamma
    output_image = cv2.addWeighted(src1=src_1, alpha=0.5, src2=src_2, beta=0.8, gamma=0)

    output_dict = {"heatmap": output_image.tolist()}
    return jsonify(output_dict)


if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0')  # run app in debug mode on port 5000
