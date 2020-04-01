# Car Classification using Transfer Learning in TensorFlow 2.x

This repository containes code and documentation for a series of blog posts I wrote together with [Stephan Müller](https://github.com/mueller-stephan) and [Dominique Lade](https://github.com/DominiqueLade) for our [STATWORX blog](https://www.statworx.com/de/blog).

The series was originally inspired by this [reddit post](https://www.reddit.com/r/MachineLearning/comments/ek5zwv/p_64000_pictures_of_cars_labeled_by_make_model/?utm_source=share&utm_medium=ios_app&utm_name=iossmf). If you want to reproduce the results, please find the data available [here](https://drive.google.com/file/d/1TQQuT60bddyeGBVfwNOk6nxYavxQdZJD/view) or alternatively go to the original [GitHub repo](https://github.com/nicolas-gervais/predicting-car-price-from-scraped-data/tree/master/picture-scraper).   

## How to use this Repo?
1. Download/Clone this repo (`git clone https://github.com/fabianmax/car-classification.git`)
2. Copy Images into Data/Images
3. Copy Model into Data/Model
4. Execute docker-compose file (`docker-compose up`)
5. Open http://localhost:8050/ and start playing! (`open http://localhost:8050/`)

Note: At the end your folder structure should be similar to this one:

```
.
+-- car_classifier
+-- dashboard
+-- data   
|   +-- images
|       +-- carbrand1_carmodel1_...._.jpg
|       +-- carbrand2_carmodel2_...._.jpg
|       +-- carbrand3_carmodel3_...._.jpg
+-- model
|   +-- saved_model.pb
|   +-- classes.pickle
|   +-- variables
|      +-- variables.index
|       +-- variables.data-00001-of-00002
|       +-- variables.data-00000-of-00002
|-- ...
```

## Part 1: Transfer Learning using ResNet50V2 in TensorFlow

In this blog, we have applied transfer learning using the ResNet50V2 to classify the car model from images of cars. Our model achieves 70% categorical accuracy over 300 classes. We found unfreezing the entire base model and using a small learning rate to achieve the best results.

Link to full blog post on STATWORX.com (coming soon)  
[Link to full blog post in this repo](https://github.com/fabianmax/car-classification/blob/master/blog/Blog_Part_1_Transfer_Learning_with_ResNet.md)

## Part 2: Deploying TensorFlow Models in Docker using TensorFlow Serving

In this blog post, we have served a TensorFlow model for image recognition using TensorFlow Serving. To do so, we first saved the model using the SavedModel format. Next, we started the TensorFlow Serving server in a Docker container. Finally, we showed how to request predictions from the model using the API endpoints and a correct specified request body.

Link to full blog post on STATWORX.com (coming soon)  
[Link to full blog post in this repo](https://github.com/fabianmax/car-classification/blob/master/blog/Blog_Part_2_Deploying_TensorFlow_Models_in_Docker_using_TensorFlow_Serving.md)

## Part 3: Explainability of Deep Learning Models with Grad-CAM

We discussed multiple approaches to explain CNN classifier outputs. We introduced Grad-CAM in detail by discussing the code and looking at examples for the car model classifier. Most notably, the discriminatory regions highlighted by the Grad-CAM procedure are always focussed on the car and never on the backgrounds of the images. The result shows that the model works as we expect and indeed uses specific parts of the car to discriminate between different models.

Link to full blog post on STATWORX.com (coming soon)   
[Link to full blog post in this repo](https://github.com/fabianmax/car-classification/blob/master/blog/Blog_Part_3_Explainable_AI_for_Computer_Vision.md)

## Part 4: Integrating Deep Learning Models with Dash

*coming soon*
