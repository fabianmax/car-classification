from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from random import choice, choices, random
from time import sleep
from typing import List

import requests

from .labels import LABELS

IMAGE_URL_INTERNAL = 'http://nginx/Raw/'
IMAGE_URL_EXTERNAL = 'http://localhost:1234/Raw/'
PREDICTION_URL_INTERNAL = 'http://tf_serving'


@dataclass
class GameData:
    items: List[Item] = field(init=False)
    current_round = 0
    max_rounds: int = 3
    validation_error: bool = False

    def __post_init__(self) -> None:
        #self.items = self.get_fake_data()
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
            print(image_name)
            image_parts = str.split(image_name, '_')
            return ItemLabel(image_parts[0], image_parts[1])

        ground_truth = map(extract_ground_truth, image_list)

        for image, truth in zip(image_list, ground_truth):
            prediction_ai = []

            for _ in range(5):
                label = self.get_fake_label()
                label.certainty = random()
                prediction_ai.append(label)

            items.append(Item(IMAGE_URL_EXTERNAL + image, '', prediction_ai, truth))

        return items

    def get_fake_data(self) -> List[Item]:
        items = []

        pictures_raw = (Path().cwd() / Path('assets/Raw')).glob('*.jpg')
        pictures_explained = (Path().cwd() / Path('assets/Explained')).glob('*.png')

        for raw, explained in zip(pictures_raw, pictures_explained):
            #for i in range(self.max_rounds + 1):
            raw = Path('Raw') / raw.parts[-1]
            explained = Path('Explained') / explained.parts[-1]

            #raw = Path('Raw') / choice(pictures_raw).parts[-1]
            #explained = Path('Explained') / choice(pictures_explained).parts[-1]

            ground_truth = self.get_fake_label()
            prediction_ai = []

            for _ in range(5):
                label = self.get_fake_label()
                #label = ground_truth
                label.certainty = random()
                prediction_ai.append(label)

            items.append(Item(raw, explained, prediction_ai, ground_truth))

        return items

    def get_fake_label(self) -> ItemLabel:
        brand = choice(list(LABELS.keys()))
        model = choice(LABELS[brand])

        return ItemLabel(brand, model)


@dataclass
class Item:
    picture_raw: Path
    picture_explained: Path
    prediction_ai: List[ItemLabel]
    ground_truth: ItemLabel
    prediction_user: ItemLabel = field(init=False)


@dataclass
class ItemLabel:
    brand: str
    model: str
    certainty: float = 1

    def __eq__(self, other) -> bool:
        if other.__class__ is not self.__class__:
            return NotImplemented
        return (self.brand, self.model) == (other.brand, other.model)
