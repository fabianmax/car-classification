
from car_classifier.preparation.pipeline import construct_dataset
from car_classifier.preparation.utils import show_batch

INPUT_DATA_DIR = 'data/raw/'
BATCH_SIZE = 32

data = construct_dataset(input_path=INPUT_DATA_DIR, batch_size=BATCH_SIZE)

show_batch(data)

data = construct_dataset(input_path=INPUT_DATA_DIR, batch_size=BATCH_SIZE, one_hot_encoding=True)

show_batch(data)