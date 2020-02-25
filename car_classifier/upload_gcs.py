import glob
from car_classifier.utils import GoogleCloudStorage

# Init custom class for uploading
gcs = GoogleCloudStorage()

# Files to upload incl. paths
files_to_upload = glob.glob('data/raw/*.jpg')

# Upload to GCS bucket 'car-classifier'
gcs.upload_files(files_to_upload)

