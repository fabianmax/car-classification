## Interpretability of Deep Learning Models with Grad-CAM

In the [first post]() (TODO: link), we built a model using transfer learning to classify the car model given an image of a car. In the [second post]() (TODO: link), we showed how TensorFlow Serving can be used to deploy a TensorFlow model using the car model classifier as an example. We dedicate this third post to another important aspect of deep learning and machine learning in general: interpretability of model predictions. We will briefly talk about popular methods that can be used to explain and interpret CNN predictions. We will then explain Grad-CAM, a gradient-based method, in depth by going through an implementation step by step. We will then show the results obtained by our Grad-CAM implementation for the car model classifier.

### Methods for explaining CNN outputs for images

Various methods to explain CNN outputs exist in the literature. Straight forward approaches are e.g. visualizing the activations in different layers to get a feel for the extracted features or using the vanilla gradients of the classes' output w.r.t. the input image to derive a heatmap for the input pixel importances. These two methods and the occlusion sensitivity method introduced below are implemented for TensorFlow 2.x in [tf-explain](https://github.com/sicara/tf-explain#vanilla-gradients). We will now briefly talk about the ideas of three other interesting approaches.

**Occlusion Sensitivity:** This approach computes the importance of certain parts of the input image by reevaluating the model's prediction with different parts of the input image hidden. Parts of the image are hidden iteratively by replacing them by grey pixels. The weaker the prediction gets with a part of the image hidden, the more important this part is for the final prediction. Based on the discriminative power of the regions of the image, a heatmap can be constructed and plotted. Our experience working with occlusion sensitivity is that the procedure often yields less confined discriminatory regions as the methods described below.

**CNN Fixations:** Another interesting approach called CNN Fixations was introduced in [this paper](https://arxiv.org/abs/1708.06670). The idea is to backtrack which neurons where important in each layer, given the activations from the forward pass and the network weights. The neurons with large influence are refered to as fixations. This approach thus allows to find the important regions for obtaining the result without the need for any recomputation (as e.g. in the case of occlusion sensitivity above, where multiple predictions must be made). The procedure can be described as follows: The node corresponding to the class is chosen as the fixation in the output layer. Then, the fixations for the previous layer are computed by computing which of the nodes have the most impact on the next higher level's fixations determined in the last step. The node importance is computed by multiplying activations and weights. If you are interested in the details of the procedure, check out [the paper](https://arxiv.org/abs/1708.06670) or the corresponding [github repo](https://github.com/val-iisc/cnn-fixations). This backtracking is done untill the input image is reached, yielding a set of pixels with large discriminatory power. An example from the paper is shown below.

!["CNN Fixation"](CNNFixation.png)

**CAM:** Introduced in [this paper](https://arxiv.org/abs/1512.04150), class activation mapping (CAM) is a procedure to find the discriminative region(s) for a CNN prediction by computing class activation maps. A large drawback of this procedure that it requires the network to use global average pooling (GAP) as the last step before the prediction layer. It thus is not possible to apply this approach to general CNNs. An example is shown in the figure below (taken from the [CAM paper](https://arxiv.org/abs/1512.04150)):

!["CAM in action"](CAM_orig_paper.png)

The class activation map assigns an importance to every position (x, y) in the last convolutional layer by computing the linear combination of the activations, weighted by the corresponding output weights for the observed class (Australian terrier in the example above). The resulting class activation mapping is then upsampled to the size of the imput image. This is depicted by the heat map above. Due to the architecture of CNNs, the activation e.g. in the top left for any layer is directly related to the top left of the input image. This is why we can conclude which regions of the input are important by only looking at the last CNN layer.

The Grad-CAM procedure we will discuss in detail below is a generalization of CAM. Grad-CAM can be applied to networks with general CNN architectures, also ones containing multiple fully connected layers at the output.

### Grad-CAM

Grad-CAM extends the applicability of the CAM procedure by incorporating gradient information. Specifically, the gradient of the loss w.r.t. the last convolutional layer determines the weight for each of its channels. As in the CAM procedure above, the further steps then are to compute the weighted sum of the activations and then upsampling the result to the image size so we can plot the original image with the obtained heatmap. We will now show and discuss the code that can be used to run Grad-CAM. The full code is available [here](https://github.com/fabianmax/car-classification/blob/master/car_classifier/grad_cam.py) on GitHub.

```python
import pickle
import tensorflow as tf
import cv2
from car_classifier.modeling import TransferModel

# Load list of targets
file = open('.../classes.pickle', 'rb')
classes = pickle.load(file)

# Load model
model = TransferModel('ResNet', INPUT_SHAPE, classes=classes)
model.load('...')

# Gradient model, outputs tuple with:
# - output of conv layer
# - output of head layer
grad_model = tf.keras.models.Model([model.model.inputs],
                                   [model.model.get_layer('conv5_block3_3_conv').output, model.model.output])

# Run model and record outputs, loss, and gradients
with tf.GradientTape() as tape:
    conv_outputs, predictions = grad_model(img)
    loss = predictions[:, label_idx]

# Output of conv layer
output = conv_outputs[0]

# Gradients of loss wrt. conv layer
grads = tape.gradient(loss, conv_outputs)[0]

# Guided Backprop (elimination of negative values)
gate_f = tf.cast(output > 0, 'float32')
gate_r = tf.cast(grads > 0, 'float32')
guided_grads = gate_f * gate_r * grads

# Average weight of filters
weights = tf.reduce_mean(guided_grads, axis=(0, 1))

# Class activation map (cam)
# Multiply values of conv filters with gradient weights
cam = np.zeros(output.shape[0: 2], dtype=np.float32)
for i, w in enumerate(weights):
    cam += w * output[:, :, i]
# cam = tf.reduce_sum(output * weights, axis=2)

# Rescale to org image size and min-max scale
cam = cv2.resize(cam.numpy(), (224, 224))
cam = np.maximum(cam, 0)
heatmap = (cam - cam.min()) / (cam.max() - cam.min())
```

* The first step is to load an instance of the model we want to explain the predictions of.
* Then, we create a new `keras.Model` instance that has two outputs: The actiations of the last CNN layer (`'conv5_block3_3_conv'`) and the original model output.
* Next, [`tf.GradientTape`](https://www.tensorflow.org/api_docs/python/tf/GradientTape) is set up and applied to record the gradients for the `conv_outputs`. Here, `img` is an input image of shape (1, 224, 224, 3), preprocessed with the `resnetv2.preprocess_input` method and `label_idx` is the index corresponding to the label we want to find the discriminatory regions for.
* In a further step, guided backdrop is applied. Only values for the gradients are kept where both the activations and the gradients are positive. This essentially means restricting the attention to the actiations which positively contribute to the wanted output prediction.
* The `weights` are computed by averaging the obtained guided gradients for each channel.
* The class activation map `cam`  then is computed as the weighted average of the channel activations (`output`). The method containing the for loop above helps understanding what the function does in detail. A less straight forward but more efficient way to implement the CAM-computation is to use `tf.reduce_mean` and is shown in the commented line below the loop implementation.
* Finally, the resampling (resizing) is done using OpenCV2's `resize` method and the heatmap is rescaled to contain values in [0, 1] for plotting.

A version of Grad-CAM is also implemented in [tf-explain](https://github.com/sicara/tf-explain#vanilla-gradients).

### Examples

We now use the Grad-CAM implementation to interpret and explain the predictions of the `TransferModel` for car model classification. We start by looking at car images taken from the front.

!["Frontal examples"](front.png)*Grad-CAM for car images from the front*

The red regions highlight the most important discriminatory regions, the blue region the least important. We can see that, for images from the front, the CNN focusses on the car's grille and the region containing the logo. If the car is slightly tilted, the focus is shifted more to the edge of the car. This is also the case for slightly tilted images from the back of cars, as can be seen in the middle image below.

!["Frontal examples"](back.png)*Grad-CAM for car images from the back*

For car images from behind, the most important discriminatory region is near the number plate. As mentioned above, for cars looked at from an angle, the closest corner has the highest discriminatory power.

!["Frontal examples"](side.png)*Grad-CAM for car images from the side*

When looking at images from the side, we notice the discriminatory region is restricted to the bottom half of the cars. Again, the angle the car image was taken from determines the shift of the region towards the front or back corner.

In general, the most important fact is that the discriminative areas are always confined to parts of the cars. There are no images where the background has high discriminatory power. Looking at the heatmaps and the associated discriminative regions can be used as a sanity check for the model.

### Deployment / API?

### Conlusion

We discussed multiple approaches to explain CNN classifier outputs. We introduced Grad-CAM in detail by discussing the code and looking at examples for the car model classifier. Most notably, the discriminatory regions highlighted by the Grad-CAM procedure are always focussed on the car and never on the backgrounds of the images. The result shows that the model works as we expect and indeed uses specific parts of the car to discriminate between different models.

In the last part of the series, we will show how the car classifier can be built in to an web application using [Dash](https://plotly.com/dash/). See you soon!

[author class="mtl" title="Über den Autor"]