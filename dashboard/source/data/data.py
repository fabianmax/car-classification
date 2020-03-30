from __future__ import annotations

import json
from dataclasses import dataclass, field
from os import getenv
from random import choices
from typing import List
from urllib.parse import quote

import numpy as np
import requests
from imageio import imread

from .labels import CLASSES

is_in_docker = getenv('IS_IN_DOCKER', False)
if not is_in_docker:
    print("Dashboard is running locally")
    IMAGE_URL_INTERNAL = IMAGE_URL_EXTERNAL = 'http://localhost:1234/Raw/'
    PREDICTION_URL_INTERNAL = 'http://localhost:8501/v1/models/resnet_unfreeze_all_filtered:predict'

else:
    print("Dashboard is running inside a docker container")
    IMAGE_URL_INTERNAL = 'http://nginx/Raw/'
    IMAGE_URL_EXTERNAL = 'http://localhost:1234/Raw/'
    PREDICTION_URL_INTERNAL = 'http://tf_serving:8501/v1/models/resnet_unfreeze_all_filtered:predict'


@dataclass
class GameData:
    items: List[Item] = field(init=False)
    current_round = 0
    max_rounds: int = 3
    validation_error: bool = False

    def __post_init__(self) -> None:
        self.items = self.get_raw_data()

    def get_raw_data(self) -> List[Item]:
        items = []

        # Send a requests to the static web server to retrieve a list with the content
        folder_content = requests.get(IMAGE_URL_INTERNAL).json()

        # Extract a list with all files in the folder
        file_list = [file['name'] for file in folder_content if file['type'] == 'file']

        # Sample randomly max rounds images from the list
        image_list = choices(file_list, k=self.max_rounds + 1)

        # Extract ground truth from the file name
        def extract_ground_truth(image_name: str) -> ItemLabel:
            image_parts = str.split(image_name, '_')
            return ItemLabel(image_parts[0], image_parts[1])

        ground_truth = map(extract_ground_truth, image_list)

        for image, truth in zip(image_list, ground_truth):
            prediction_ai = self.get_ai_prediction(image)

            items.append(
                Item(IMAGE_URL_EXTERNAL + image, 'TODO: explained image', prediction_ai,
                     truth))

        return items

    def get_ai_prediction(self, image_name: str) -> List[ItemLabel]:
        # Download Picture
        img_url = IMAGE_URL_INTERNAL + quote(image_name)
        img = imread(img_url)

        # Get Prediction from TF Serving
        # Preprocess and reshape data
        img = img.astype(np.float32)
        img /= 127.5
        img -= 1.
        img = img.reshape(-1, *img.shape)

        # Send data as list to TF serving via json dump
        request_url = PREDICTION_URL_INTERNAL
        request_body = json.dumps({
            "signature_name": "serving_default",
            "instances": img.tolist()
        })
        request_headers = {"content-type": "application/json"}
        json_response = requests.post(request_url,
                                      data=request_body,
                                      headers=request_headers)
        response_body = json.loads(json_response.text)
        predictions = response_body['predictions']

        top_predictions = []
        for idx in np.argpartition(predictions[0], -5)[-5:]:
            label = CLASSES[idx]
            label_comp = label.split('_')
            brand = label_comp[0]
            model = label_comp[1]
            certainty = predictions[0][idx]

            top_predictions.append(ItemLabel(brand, model, certainty))

        top_predictions = sorted(top_predictions,
                                 key=lambda x: x.certainty,
                                 reverse=True)

        return top_predictions


@dataclass
class Item:
    picture_raw: str
    picture_explained: str
    prediction_ai: List[ItemLabel]
    ground_truth: ItemLabel
    prediction_user: ItemLabel = field(init=False)


@dataclass
class ItemLabel:
    brand: str
    model: str
    certainty: float = 1

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, ItemLabel):
            return NotImplemented
        return (self.brand, self.model) == (other.brand, other.model)
