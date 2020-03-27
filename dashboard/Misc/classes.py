"""Convert the pickle file with the labels into a dictionary with the
car brand as key and a list of the models as value
"""

import json

import pandas as pd

# Read Pickle File
df = pd.read_pickle('classes.pickle')

# Split Column with Labels by underscore
df = df.classes.str.split("_", expand=True)

# Create the dictionary with brand -> list of models
final_data = dict()

for brand in df[0].unique():
    final_data[brand] = df[df[0] == brand][1].tolist()

# Save the dictionary as a python file
with open("labels.py", "w") as dict_file:
    dict_file.write('LABELS = ' + json.dumps(final_data))

print('Finished processing!')
