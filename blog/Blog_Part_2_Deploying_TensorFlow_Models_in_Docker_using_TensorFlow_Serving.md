# Deploying TensorFlow Models in Docker using TensorFlow Serving

In the [first post]() (TODO link) of this series, we discussed transfer learning and built a model for car model classification. In this blog post, we will discuss the problem of model deployment, using the `TransferModel` introduced in the first post as an example. 

In practice, a model is of no use if there is no simple way it can be interacted with. In other words: We need an API for our models. TensorFlow Serving has been developed to provide these functionalities for TensorFlow models. In this blog post, we will show how a TensorFlow Serving server can be launched in a Docker container and how we can interact with the server using HTTP requests. If you are new to Docker, we recommend working through Docker's [tutorial](https://docker-curriculum.com/) prior to reading this post. If you want to look at an example of deployment in Docker, we recommend reading Oli's [blog post](https://www.statworx.com/de/blog/running-your-r-script-in-docker/) describing how an R-script can be run in Docker. We start by giving an overview of TensorFlow Serving.

## Introduction to TensorFlow Serving

TensorFlow Serving is TensorFlow's serving system, designed to enable deployment of various models using a uniform API. Using the abstraction of Servables, which are basically objects clients use to perform computations, it is possible to serve multiple versions of deployed models. This enables for example that a new version of a model can be uploaded while the previous version is still available to clients. Looking at the bigger picture, so called Managers are responsable for handling the life-cycle of Servables, that means loading, serving and unloading them. In this post, we will show how a single model version can be deployed. In the code examples below, we will show both how a server can be started in a Docker container and how the Predict API can be used to interact with it. To read more about TensorFlow Serving, we refer to the [TensorFlow website](https://www.tensorflow.org/tfx/guide/serving).

## Implementation

We will now discuss the following three steps required to deploy the model and to send requests.

* Save a model in correct format and folder structure using TensorFlow SavedModel
* Run a Serving server inside a Docker container
* Interact with the model using REST requests

### Saving TensorFlow Models

If you didn't read this series' first post, we briefly summarize the most important points needed to understand the code below: 

The `TransferModel.model` is a `tf.keras.Model` instance, so it can be saved using `Model`'s built-in `save` method. Further, as the model was trained on web-scraped data, the class labels can change when re-scraping the data. We thus store the index-class mapping when storing the model in `classes.pickle`. TensorFlow Serving requires the model to be stored in the [SavedModel format](https://www.tensorflow.org/guide/saved_model). When using `tf.keras.Model.save`, the path must be a folder name, else the model will be stored in another format (e.g. HDF5) which is not compatible with TensorFlow Serving. Below, `folderpath` contains the path of the folder we want to store all model relevant information in. The SavedModel is stored in `folderpath/model` and the class mapping is stored as `folderpath/classes.pickle`.

```python
def save(self, folderpath: str):
    """
    Save the model using tf.keras.model.save

    Args:
        folderpath: (Full) Path to folder where model should be stored
    """

    # Make sure folderpath ends on slash, else fix
    if not folderpath.endswith("/"):
        folderpath += "/"

    if self.model is not None:
        os.mkdir(folderpath)
        model_path = folderpath + "model"
        # Save model to model dir
        self.model.save(filepath=model_path)
        # Save associated class mapping
        class_df = pd.DataFrame({'classes': self.classes})
        class_df.to_pickle(folderpath + "classes.pickle")
    else:
        raise AttributeError('Model does not exist')
```

### Start TensorFlow Serving in Docker Container

Having saved the model to the disk, the next step is to start the TensorFlow Serving server. Fortunately, there is an easy-to-use Docker container available. The first step is therefore pulling the TensorFlow Serving image from DockerHub. This can be done in the terminal using the command `docker pull tensorflow/serving`. 

Then we can use the code below to start a TensorFlow Serving container. It runs the shell command for starting a container. The options set in the `docker_run_cmd` are the following: 

- The serving image exposes port 8501 for the REST API which we will use later to send requests. Thus we map the host port 8501 to the container's 8501 port using `-p`. 
- Next, we mount our model to the container using `-v`. It is essential that the model is stored in a versioned folder (here MODEL_VERSION=1) else the serving image will not find the model. `model_path_guest` thus must be of the form `<path>/<model name>/MODEL_VERSION`, where `MODEL_VERSION` is an integer.
- Using `-e`, we can set the environment variable `MODEL_NAME` to our model's name.
- The `--name tf_serving` option is only needed to assign a specific name to our new docker container.

If we try to run this file twice in a row, the docker command will not be executed the second time, as a container with the name `tf_serving` already exists. To avoid this problem, we use `docker_run_cmd_cond`. Here, we first check if a container with this specific name already exists and is running. If yes, we leave it, if no, we check if an exited version of the container exists. If yes, it is deleted and a new container is started, else a new one is started directly. 

```python
import os

MODEL_FOLDER = 'models'
MODEL_SAVED_NAME = 'resnet_unfreeze_all_filtered.tf'
MODEL_NAME = 'resnet_unfreeze_all_filtered'
MODEL_VERSION = '1'

# Define paths on host and guest system
model_path_host = os.path.join(os.getcwd(), MODEL_FOLDER, MODEL_SAVED_NAME, 'model')
model_path_guest = os.path.join('/models', MODEL_NAME, MODEL_VERSION)

# Container start command
docker_run_cmd = f'docker run ' \
                 f'-p 8501:8501 ' \
                 f'-v {model_path_host}:{model_path_guest} ' \
                 f'-e MODEL_NAME={MODEL_NAME} ' \
                 f'-d ' \
                 f'--name tf_serving ' \
                 f'tensorflow/serving'

# If container is not running, create a new instance and run it
docker_run_cmd_cond = f'if [ ! "$(docker ps -q -f name=tf_serving)" ]; then \n' \
                      f'   if [ "$(docker ps -aq -f status=exited -f name=tf_serving)" ]; then 														\n' \
                      f'   		docker rm tf_serving \n' \
                      f'   fi \n' \
                      f'   {docker_run_cmd} \n' \
                      f'fi'

# Start container
os.system(docker_run_cmd_cond)
```
Instead of mounting the model from our local disk using the `-v` flag in the docker command, we could also copy the model into the docker image, so the model could be served simply by running a container and specifying the port assignments. It is important to note that, in this case, the model needs to be saved using the folder structure `folderpath/<model name>/1`, as explained above. If this is not the case, TensorFlow Serving will not find the model. We will not go into further detail here. If you are interested in deploying your models in this way, we refer to [this guide](https://www.tensorflow.org/tfx/serving/docker#creating_your_own_serving_image) on the TensorFlow website.

### REST Request

Since the model is now served and ready to use, we need a way to interact with it. TensorFlow Serving provides two options to send requests to the server: [gRCP](https://grpc.io/) and REST API, both exposed at different ports. In the following code example, we will use REST to query the model.

First, we load an image from the disk for which we want a prediction. This can be done using TensorFlow's `image` module. Next, we convert the image to a numpy array using the `img_to_array`-method. The next and final step is very important: since we preprocessed the training image before we trained our model (e.g. normalization), we need to apply the same transformation on image we want to predict. The handy `preprocess_input` function makes sure that all necessary transformations are applied to our image.

```python
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.resnet_v2 import preprocess_input

# Load image
img = image.load_img(path, target_size=(224, 224))
img = image.img_to_array(img)

# Preprocess and reshape data
img = preprocess_input(img)
img = img.reshape(-1, *img.shape)
```

TensorFlow Serving's RESTful API offers several endpoints. In general, the API accepts post requests following this structure:

```
POST http://host:port/<URI>:<VERB>

URI: /v1/models/${MODEL_NAME}[/versions/${MODEL_VERSION}]
VERB: classify|regress|predict
```

For our model, we can use the following URL for predictions: `http://localhost:8501/v1/models/resnet_unfreeze_all_filtered:predict`

The port number (here 8501) is the host's port we specified above to map to the serving image's port 8501. As mentioned above, 8501 is the serving container's port exposed for the REST API. The model version is optional and will default to the latest version if omitted. 

In python, the `requests` library can be used to send HTTP requests. As stated in the [documentation](https://www.tensorflow.org/tfx/serving/api_rest), the request body for the `predict` API must be a JSON object with the below listed key-value-pairs:

- `signature_name` - serving signature to use (for more information see the [documentation](https://www.tensorflow.org/tfx/serving/signature_defs))
- `instances` - model input in row format

```python
import json
import requests

# Send image as list to TF serving via json dump
request_url = 'http://localhost:8501/v1/models/resnet_unfreeze_all_filtered:predict'
request_body = json.dumps({"signature_name": "serving_default", "instances": img.tolist()})
request_headers = {"content-type": "application/json"}
json_response = requests.post(request_url, data=request_body, headers=request_headers)
response_body = json.loads(json_response.text)
predictions = response_body['predictions']

# Get label from prediction
y_hat_idx = np.argmax(predictions)
y_hat = classes[y_hat_idx]
```

The response body will also be a JSON object with a single key called `predictions`. Since we get for each row in instances the probability for all 300 classes, we use `np.argmax` to return the most likely class. Alternatively, we could have used the higher-level classify API.

### Conclusion

In this blog post, we have served a TensorFlow model for image recognition using TensorFlow Serving. To do so, we first saved the model using the SavedModel format. Next, we started the TensorFlow Serving server in a Docker container. Finally, we showed how to request predictions from the model using the API endpoints and a correct specified request body. In the next post, we will show how to explain model predictions using a method called Grad-CAM.

[author class="mtl" title="Über den Autor"]