# How to Develop a Currency Detection Model using Azure Machine Learning

This repository contains the code for the blogpost: [How to Develop a Currency Detection Model using Azure Machine Learning](https://blogs.technet.microsoft.com/machinelearning/2018/05/01/how-to-develop-a-currency-detection-model-using-azure-machine-learning/)

# Introduction

&quot;How do we teach a machine to see?&quot;

[Seeing AI](https://www.youtube.com/watch?v=bqeQByqf_f8) is an exciting research project that harnesses the power of Artificial Intelligence to open the visual world and describe nearby people, text, currency, color and objects with spoken audio. It is designed for the blind and low vision community, to understand more about who and what is around them. Today, the iOS smartphone app has empowered users to complete over 5 million tasks unassisted, including many &quot;first time in life&quot; experiences for the blind community, like take and posting photos of friends on Facebook, recognizing products independently at a store, reading kids homework and more. To learn more about Seeing.AI – visit [https://www.microsoft.com/en-us/seeing-ai/](https://www.microsoft.com/en-us/seeing-ai/).

One of the most common needs among the blind community is to recognize paper currency. Currency notes are usually inaccessible, as they are hard to recognize purely through tactical feeling. To solve this scenario, the Seeing AI team built a [real time currency recognizer](https://youtu.be/S8XpZpeUYnU) which can instantly speak the denomination in view, with high precision (in under 25 milliseconds). Since the target user often does not have perception of whether the currency note is in the camera view or not, having a real time spoken experience acts as a feedback to help user frame it until it&#39;s clearly visible as well as not too close or too dark.

In this blog post, we are excited to share with you the secrets behind building and training a currency prediction model, as well as deploy to the intelligent cloud and intelligent edge.

You will learn how to:

- Build a deep learning model on small data using transfer learning. Specifically, we will develop the model using Keras, Deep Learning Virtual Machine (DLVM), and Visual Studio Tools for AI.
- Create a scalable API with 1 line of code (With Azure Machine Learning)
- Export a mobile optimized model (With CoreML)

# Boost AI productivity with the right tools

When developing deep learning models, using the right AI tools can boost your productivity - A pre-configured VM for deep learning development, and a familiar IDE which integrates deeply with the deep learning environment.

## Deep Learning Virtual Machines (DLVM)

[Deep Learning Virtual Machine](https://docs.microsoft.com/en-us/azure/machine-learning/data-science-virtual-machine/deep-learning-dsvm-overview) (DLVM) enables users to jumpstart their deep learning projects. The DLVM is pre-packaged VM with lots of useful tools (e.g. installed GPU drivers, availability of popular deep learning frameworks, etc.) that can facilitate any deep learning project. Using DLVM, a data scientist can be productive in minutes.

## Visual Studio Tools for AI

[Visual Studio Tools for AI](https://www.visualstudio.com/downloads/ai-tools-vs/) is an extension that supports deep learning frameworks including  [Microsoft Cognitive Toolkit (CNTK)](https://www.microsoft.com/en-us/cognitive-toolkit/),  [TensorFlow](https://www.tensorflow.org/),  [Keras](https://keras.io/),  [Caffe2](https://caffe2.ai/) and more. It provides nice language features such as IntelliSense, as well as debugging capabilities such as TensorBoard integration. These features make it an ideal choice for cloud-based AI development.

Figure 4 Visual Studio Tools for AI

# Building the currency detection model

In this section you will learn how to build, train and deploy a currency detection model to Azure and the intelligent edge.

## Dataset preparation and pre-processing

In this section, we will share how you can create the dataset needed for training the model.

In our case, the dataset consists of 15 classes. These include 14 classes denoting the different denominations (inclusive both the front and back of the currency note), and an additional class denoting &quot;background&quot;. Each class has around 250 images (with notes placed in various places and in different angles, see below), You can easily create the dataset needed for training in half an hour with your phone.

For the &quot;background&quot; class, you can use images from [ImageNet Samples](https://planspace.org/20170430-sampling_imagenet/). We put 5 times more images in the background class than the other classes, to make sure the deep learning algorithm does not learn a pattern.

Once you have created the dataset, and trained the model, you should be able to get an accuracy of approximately 85% (using the transfer learning and data augmentation techniques mentioned below). If you want to improve the performance, refer to the &quot;Further discussions&quot; section below.

Figure 2 Data Organization Structure

Below is an illustration for one of the samples in the training dataset.  We experimented with different options and find that the training data should contain a few images described below, so the model can get decent performance when applied in real-life.

1. The currency note should occupy at least 1/6 of the whole image
2. The currency note should be displayed from different angles in the image
3. The currency note should be present in various locations in the image (top left corner, middle, bottom right corner, etc.)
4. There should be some foreground objects covering part of the currency (no more than 40% though)
5. The background should be as diversified as possible

Figure 3 One of the pictures in the dataset

## Choosing the right model

Tuning deep neural architectures to strike an optimal balance between accuracy and performance has been an area of active research for the last few years. This becomes even more challenging when you need to deploy the model to mobile devices and still ensure it is high-performing. For example, when building the Seeing AI applications, the currency detector model need to run locally on the cell phone. Inference on the phone need to have low latency to ensure the best user experience and without sacrificing accuracy.

One of the most important metrics used for measuring the number of operations performed, during model inference, is called **multiply-adds** (abbreviated as MAdd). The trade-offs between speed and accuracy across different models are shown in Figure 1.

Figure 1 Accuracy vs time, the size of each blob represents the number of parameters. Source: [https://github.com/tensorflow/models/tree/master/research/slim/nets/mobilenet](https://github.com/tensorflow/models/tree/master/research/slim/nets/mobilenet)

In Seeing AI, we choose to use **MobileNet** because it&#39;s fast enough on cell phones and provides decent performance based on our empirical experiments.

## Build and train the model

Since the data set is small (only 250 images per class), we use two techniques to solve the problem:

1. Doing transfer learning with pre-trained models on large datasets
2. Using data augmentation techniques

### Transfer Learning

Transfer learning is a machine learning method where you start off using a pre-trained model, adapt and fine-tune it for other domains. For use cases such as Seeing AI, because the dataset is not large enough, starting with a pre-trained model, and further fine-tuning the model can reduce training time, and alleviate possible overfitting.

In practice, using transfer learning often requires you to &quot;freeze&quot; a few top layers&#39; weights of a pre-trained model, then let the rest of the layers be trained normally (so back-propagation process can change their weights). Using Keras for transfer learning is quite easy – just set the trainable parameter of the layers which you want to freeze to False, and Keras will stop updating the parameters of those layers, while still back propagate the weights of the rest of the layers:

Figure 5 Transfer learning in Keras

### Data Augmentation

Since we don&#39;t have enough input data, another approach is to reuse existing data as much as possible. For images, common techniques include shifting the images, zooming in or out of the images, rotating the images, etc., which could be easily done in Keras:

Figure 6 Data Augmentation

## Deploy the model

### Deploy the model to the intelligent edge

For applications like Seeing AI, we want to run the models locally, so the application can always be used even when the internet connection is poor. Exporting a Keras model to CoreML, which can be consumed by iOS application, can be achieved by coremltools:

model\_coreml = coremltools.converters.keras.convert(&quot;currency\_detector.h5&quot;, image\_scale = 1./255)

### Deploy the model to Azure as a REST API

In some other cases, data scientists want to deploy a model and expose an API which can be further used by the developer team. However, releasing the model as a REST API is always challenging for enterprise scenarios, and [Azure Machine Learning services](https://azure.microsoft.com/en-us/services/machine-learning-services/) enables data scientists to easily deploy their model on the cloud in a secure and scalable way. To operationalize your model using Azure Machine Learning, you can leverage the Azure Machine Learning Command-line interface and specify the required configurations using [AML operationalization module](https://docs.microsoft.com/en-us/azure/machine-learning/preview/tutorial-classifying-iris-part-3), like below:

az ml service create realtime -f score.py --model-file currency\_detector.pkl -s service\_schema.json -n currency\_detector\_api -r python --collect-model-data true -c aml\_config\conda\_dependencies.yml

The end to end architecture is as below:

Figure 7 End to end architecture on developing a currency detection model and deploy to both cloud and intelligent edge devices

# Further discussions

## Even faster models

Recently, a newer version of MobileNet was released, called [MobileNetV2](https://arxiv.org/abs/1801.04381). The test done by the author shows that the newer version is 35% faster than the V1 version, when running on a Google Pixel phone using CPU (200ms vs. 270ms) at the same accuracy. This enables a more pleasant user experience.

Figure 8 Accuracy vs Latency between MobileNet V1 and V2. [Source](https://github.com/tensorflow/models/tree/master/research/slim/nets/mobilenet)

## Synthesizing more training data

To improve the performance of the models, getting more data is key. While we can collect more real-life data, we should also investigate different ways to synthesize more data.

The simple approach is to use images that are of sufficient resolution for various currencies (such as [image search result returned by Bing](https://www.bing.com/images/search?q=us+dollar&amp;FORM=HDRSC2)), transform them, and overlay on diverse images, such as [a small sample of ImageNet](https://planspace.org/20170430-sampling_imagenet/). OpenCV provides [several transformation techniques](http://opencv-python-tutroals.readthedocs.io/en/latest/py_tutorials/py_imgproc/py_geometric_transformations/py_geometric_transformations.html#geometric-transformations). These include scaling, rotation, affine transformation, and perspective transformation.

In practice, we train the algorithm on synthetic data and validate the algorithm on the real-world dataset we collected.

# Conclusion

In this blog post, we have described how you can easily build your own currency detector using deep learning virtual machines (DLVM) and Azure. We show you how you can build, train and deploy powerful deep learning models using a small dataset. And with just 1 line of code, we exported the model to CoreML. This enables you to build innovative mobile applications, running on iOS.

The code is open-source on [GitHub](https://github.com/xiaoyongzhu/SeeingAI-Currency-Detection). If you have any questions, feel free to reach out to us at xiaoyzhu@microsoft.com.

Xiaoyong, Anirudh &amp; Wee Hyong
