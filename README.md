# Car Classification using Transfer Learning in TensorFlow 2.x

This repository containes code and documentation for a series of blog posts I wrote together with [Stephan Müller](https://github.com/mueller-stephan) for our [STATWORX blog](https://www.statworx.com/de/blog).

The series was originally inspired by this [reddit post](https://www.reddit.com/r/MachineLearning/comments/ek5zwv/p_64000_pictures_of_cars_labeled_by_make_model/?utm_source=share&utm_medium=ios_app&utm_name=iossmf). If you want to reproduce the results, please find the data available [here](https://drive.google.com/file/d/1TQQuT60bddyeGBVfwNOk6nxYavxQdZJD/view) or alternatively go to the original [GitHub repo](https://github.com/nicolas-gervais/predicting-car-price-from-scraped-data/tree/master/picture-scraper).   

## Part 1: Transfer Learning using ResNet50V2 in TensorFlow

In this blog, we have applied transfer learning using the ResNet50V2 to classify the car model from images of cars. Our model achieves 70% categorical accuracy over 300 classes. We found unfreezing the entire base model and using a small learning rate to achieve the best results.

Link to full blog post on STATWORX.com (coming soon)  
[Link to full blog post in this repo](https://github.com/fabianmax/car-classification/blob/master/blog/Blog_Part_1_Transfer_Learning_with_ResNet.md)

## Part 2: Deployment of Deep Learning Models with TensorFlow Serving

In this blog post, we have served a TensorFlow model for image recognition using TensorFlow Serving. To do so, we first saved the model using the SavedModel format. Next, we started the TensorFlow Serving server in a Docker container. Finally, we showed how to request predictions from the model using the API endpoints and a correct specified request body.

Link to full blog post on STATWORX.com (coming soon)  
[Link to full blog post in this repo](https://github.com/fabianmax/car-classification/blob/master/blog/Blog_Part_2_Deploying_TensorFlow_Models_in_Docker_using_TensorFlow_Serving.md)

## Part 3: Interpretability of Deep Learning Models with Grad-CAM

*comming soon*

## Part 4: Integrating Deep Learning Models with Dash

*comming soon*
