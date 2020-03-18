import os
import pickle
import json
import requests
import numpy as np
import matplotlib.pyplot as plt

from random import sample
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.resnet_v2 import preprocess_input

"""
# Original docker command
docker run \
       -p 8501:8501 \
       -v /Users/fabianmueller/Intern/Projekte/Car-Classifier/models/resnet_unfreeze_all_filtered.tf:/models/resnet_unfreeze_all_filtered/1 \
       -e MODEL_NAME=resnet_unfreeze_all_filtered \
       --name tf_serving \
       tensorflow/serving
"""

# Global model settings
INPUT_DATA_DIR = 'data/cars_filtered_top300/'
INPUT_SHAPE = (224, 224, 3)
TARGET = 'model'

MODEL_FOLDER = 'models'
MODEL_SAVED_NAME = 'resnet_unfreeze_all_filtered.tf'
MODEL_NAME = 'resnet_unfreeze_all_filtered'
MODEL_VERSION = '1'

# Paths for volume mounting
model_path_host = os.path.join(os.getcwd(), MODEL_FOLDER, MODEL_SAVED_NAME)
model_path_guest = os.path.join('/models', MODEL_NAME, MODEL_VERSION)

# Container start command
docker_run_cmd = f'docker run ' \
                 f'-p 8501:8501 ' \
                 f'-v {model_path_host}:{model_path_guest} ' \
                 f'-e MODEL_NAME={MODEL_NAME} ' \
                 f'-d ' \
                 f'--name tf_serving ' \
                 f'tensorflow/serving'

# If container is not running, create a new instance and run it
docker_run_cmd_cond = f'if [ ! "$(docker ps -q -f name=tf_serving)" ]; then \n' \
                      f'   if [ "$(docker ps -aq -f status=exited -f name=tf_serving)" ]; then \n' \
                      f'   docker rm tf_serving \n' \
                      f'   fi \n' \
                      f'   {docker_run_cmd} \n' \
                      f'fi'

# Start container
os.system(docker_run_cmd_cond)

# All available training images
files = [file for file in os.listdir(INPUT_DATA_DIR) if file.endswith(".jpg")]
file_paths = [INPUT_DATA_DIR + file for file in files]

# Load list of targets
file = open('models/classes_all_filtered.pkl', 'rb')
classes = pickle.load(file)

# Get file
path = sample(file_paths, 1)[0]

# Get label
parts = path.split('/')[-1]
label = parts.split('_')[0] + "_" + parts.split('_')[1]

# Prepare image
img = image.load_img(path, target_size=(224, 224))
img = image.img_to_array(img)

# Show image
plt.figure()
plt.imshow(img/255)
plt.axis('off')
plt.show()

# Preprocess and reshape data
img = preprocess_input(img)
img = img.reshape(-1, *img.shape)

# Send data as list to TF serving via json dump
data = json.dumps({"signature_name": "serving_default", "instances": img.tolist()})
headers = {"content-type": "application/json"}
json_response = requests.post('http://localhost:8501/v1/models/resnet_unfreeze_all_filtered:predict', data=data, headers=headers)
predictions = json.loads(json_response.text)['predictions']

# Get label from prediction
y_hat_idx = np.argmax(predictions)
y_hat = classes[y_hat_idx]

# Print label and prediction
print(label, y_hat)
