import os
from random import sample

import matplotlib.pyplot as plt
import requests
from imageio import imread

# Global model settings
INPUT_DATA_DIR = 'Data/Images/'
INPUT_SHAPE = (224, 224, 3)

# All available training images
files = [file for file in os.listdir(INPUT_DATA_DIR) if file.endswith(".jpg")]
file_paths = [INPUT_DATA_DIR + file for file in files]

sample_path = sample(files, 1)[0]

img = imread(INPUT_DATA_DIR + sample_path)

# Send data as list to TF serving via json dump
request_url = 'http://localhost:5000/grad-cam'
request_body = {"label": sample_path, "image": img.tolist()}
request_headers = {"content-type": "application/json"}
response = requests.post(request_url, json=request_body, headers=request_headers)

heatmap = response.json()['heatmap']

plt.imshow(heatmap)
plt.show()
