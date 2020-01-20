import os
import pandas as pd

from car_classifier.preparation.utils import open_file_structure, expand_column

# Get jpg-files in raw folder
files = [file for file in os.listdir('data/raw') if file.endswith(".jpg")]

# Get file structure
file_name_components = open_file_structure('data/label_structure.txt', True)

# Create lookup table containing multiple information about images
label_df = pd.DataFrame({'file_name': files})
label_df = expand_column(label_df, 'file_name', file_name_components)

# Serialize data as pickle
label_df.to_pickle('data/prepared/label_df.pickle')
