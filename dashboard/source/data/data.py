from __future__ import annotations

from base64 import b64encode
from dataclasses import dataclass, field
from io import BytesIO
from os import getenv
from random import choices
from typing import List
from urllib.parse import quote

import matplotlib.pyplot as plt
import numpy as np
import requests
from imageio import imread

from .labels import CLASSES

# Set matplotlib to a non-interactive backend
plt.switch_backend('Agg')

# Check whether the script is running in a docker container.
# Set the URLs accordingly
is_in_docker = getenv('IS_IN_DOCKER', False)
if not is_in_docker:
    IMAGE_URL_INTERNAL = IMAGE_URL_EXTERNAL = 'http://localhost:1234/'
    PREDICTION_URL_INTERNAL = 'http://localhost:8501/v1/models/resnet_unfreeze_all_filtered:predict'
    EXPLAINABILITY_URL_EXTERNAL = 'http://localhost:5000/grad-cam'

else:
    IMAGE_URL_INTERNAL = 'http://nginx/'
    IMAGE_URL_EXTERNAL = 'http://localhost:1234/'
    PREDICTION_URL_INTERNAL = 'http://tf_serving:8501/v1/models/resnet_unfreeze_all_filtered:predict'
    EXPLAINABILITY_URL_EXTERNAL = 'http://explainability:5000/grad-cam'


@dataclass
class GameData:
    """Class to store all game related data
    """
    items: List[Item] = field(init=False)
    current_round = 0
    max_rounds: int = 4
    validation_error: bool = False

    def __post_init__(self) -> None:
        """Initialize data upon creation of the class
        """
        self.items = self.initialize_data()

    def reset(self) -> None:
        """Reset the game state
        """
        self.current_round = 0
        self.validation_error = False
        self.items = self.initialize_data()

    def initialize_data(self) -> List[Item]:
        """Initialize game data class with random selected images, the
        explainable images and the ai prediction

        Returns:
            List[Item] -- List with items; One per round
        """
        items = []

        images = self.get_images()
        ground_truth = map(self.extract_ground_truth, images)

        for image, truth in zip(images, ground_truth):
            # Download Picture from static web-server
            img_url = IMAGE_URL_INTERNAL + quote(image)
            img = imread(img_url)

            # Get AI prediction
            prediction_ai = self.get_ai_prediction(img)
            image_explainability = self.image_array_to_string(
                self.get_explainability(img, truth.brand + '_' + truth.model))

            items.append(
                Item(IMAGE_URL_EXTERNAL + image, image_explainability, prediction_ai,
                     truth))

        return items

    def extract_ground_truth(self, image_name: str) -> ItemLabel:
        """Extract car brand and car model from a image name

        Arguments:
            image_name {str} -- image name

        Returns:
            ItemLabel -- label for image
        """
        image_parts = str.split(image_name, '_')
        return ItemLabel(image_parts[0], image_parts[1])

    def image_array_to_string(self, image: np.array) -> str:
        """Converts a numpy array to a base64 encoded image.
        That is needed to pass it to a html atribute as the source value.

        Arguments:
            image {np.array} -- raw image

        Returns:
            str -- base64 encoded image
        """
        # Generate Matplotlib plot
        sizes = np.shape(image)
        fig = plt.figure()
        fig.set_size_inches(1. * sizes[0] / sizes[1], 1, forward=False)
        ax = plt.Axes(fig, [0., 0., 1., 1.])
        ax.set_axis_off()
        fig.add_axes(ax)
        ax.imshow(image)

        # Save Matplotlib image in output_image
        output_image = BytesIO()
        fig.savefig(output_image, format='png', dpi=sizes[0])

        # Close all plots
        fig.clf()
        plt.close('all')

        # Encode matplotlib plot as base64
        output_image.seek(0)  # rewind file
        encoded = b64encode(output_image.read()).decode("ascii").replace("\n", "")

        return "data:image/png;base64,{}".format(encoded)

    def get_images(self) -> List[str]:
        """From the static web-server, get a list with all images
        and sample randomly for max_rounds

        Returns:
            List[str] -- Image names
        """
        # Send a requests to the static web server to retrieve a list with the content
        folder_content = requests.get(IMAGE_URL_INTERNAL).json()

        # Extract a list with all files in the folder
        files = [file['name'] for file in folder_content if file['type'] == 'file']

        # Sample randomly max rounds images from the list
        images = choices(files, k=self.max_rounds + 1)

        return images

    def get_ai_prediction(self, img: np.array) -> List[ItemLabel]:
        """Obtain prediction from ai

        Arguments:
            img {str} -- raw image

        Returns:
            List[ItemLabel] -- Top 5 prediction from ai
        """
        # Get Prediction from TF Serving
        # Preprocess and reshape data
        img = img.astype(np.float32)
        img /= 127.5
        img -= 1.
        img = img.reshape(-1, *img.shape)

        # Send data as list to TF serving via json dump
        request_body = {"signature_name": "serving_default", "instances": img.tolist()}
        request_headers = {"content-type": "application/json"}
        response = requests.post(PREDICTION_URL_INTERNAL,
                                 json=request_body,
                                 headers=request_headers)
        predictions = response.json()['predictions']

        # Extract top 5 predictions and get the label for them
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

    def get_explainability(self, img: np.array, label: str) -> np.array:
        """Obtain explained image from API

        Arguments:
            img {np.array} -- raw image
            label {str} -- label

        Returns:
            np.array -- explained image/heatmap
        """
        body = {"label": label, "image": img.tolist()}
        header = {"content-type": "application/json"}
        response = requests.post(EXPLAINABILITY_URL_EXTERNAL, json=body, headers=header)

        heatmap = response.json()['heatmap']
        return heatmap


@dataclass
class Item:
    """Class to store items - one item per round
    """
    picture_raw: str
    picture_explained: str
    prediction_ai: List[ItemLabel]
    ground_truth: ItemLabel
    prediction_user: ItemLabel = field(init=False)


@dataclass
class ItemLabel:
    """Class to store labels
    """
    brand: str
    model: str
    certainty: float = 1

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, ItemLabel):
            return NotImplemented
        return (self.brand, self.model) == (other.brand, other.model)
