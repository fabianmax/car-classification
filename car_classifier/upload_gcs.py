import glob
from tqdm import tqdm
from google.cloud import storage


class GoogleCloudStorage:
    """
    Class for up/down-loading files to Google Cloud Storage
    """

    def __init__(self):
        self.bucket = 'car-classifier'
        self.credentials = 'resources/STATWORX-5db149736e9d.json'
        self.storage_client = storage.Client.from_service_account_json(self.credentials)

    def _upload_blob(self, source_file_name, destination_blob_name):

        bucket = self.storage_client.get_bucket(self.bucket)
        blob = bucket.blob(destination_blob_name)

        blob.upload_from_filename(source_file_name)

    def _download_blob(self, source_blob_name, destination_file_name):

        bucket = self.storage_client.get_bucket(self.bucket)
        blob = bucket.blob(source_blob_name)

        blob.download_to_filename(destination_file_name)

    def upload_files(self, files):

        if isinstance(files, str):

            source = files
            destination = files.split('/')[-1]
            self._upload_blob(source, destination)
            print(f'File {source} uploaded to {destination}')

        elif isinstance(files, list):

            source = files
            destination = [x.split('/')[-1] for x in files]

            for s, d in tqdm(zip(source, destination), total=len(source)):
                self._upload_blob(s, d)

    def download_files(self, source_files, destination_path):

        if isinstance(source_files, str):

            source = source_files
            destination = destination_path + source_files

            self._download_blob(source, destination)
            print(f'File {source} downloaded to {destination}')

        elif isinstance(source_files, list):

            source = source_files
            destination = [destination_path + x for x in source_files]

            for s, d in tqdm(zip(source, destination), total=len(source)):
                self._download_blob(s, d)


# Init custom class for uploading
gcs = GoogleCloudStorage()

# Files to upload incl. paths
files_to_upload = glob.glob('data/raw/*.jpg')

# Upload to GCS bucket 'car-classifier'
gcs.upload_files(files_to_upload)

