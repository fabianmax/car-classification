import os
import json
import requests

import matplotlib.pyplot as plt

from random import sample
from tensorflow.keras.preprocessing import image


# Global model settings
INPUT_DATA_DIR = '/Users/stephanmueller/Intern/Projekte/Car-Classifier/car-classification/car_classifier/data/' \
                 'final_filtered_split/test/'
INPUT_SHAPE = (224, 224, 3)

# All available training images
files = [file for file in os.listdir(INPUT_DATA_DIR) if file.endswith(".jpg")]
file_paths = [INPUT_DATA_DIR + file for file in files]

sample_path = sample(files, 1)[0]

# TODO: correct resizing
img = image.load_img(INPUT_DATA_DIR + sample_path, target_size=(224, 224))
img = image.img_to_array(img)

# Send data as list to TF serving via json dump
request_url = 'http://localhost:5000/grad-cam'
request_body = json.dumps({"label": sample_path, "image": img.tolist()})
request_headers = {"content-type": "application/json"}
json_response = requests.post(request_url, data=request_body, headers=request_headers)
response_body = json.loads(json_response.text)
heatmap = response_body['heatmap']

plt.imshow(heatmap)
plt.show()
