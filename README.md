# FaceVerification
Given a reference image, the project aims to detect if a new image is an image of the same person or an image of a different person.

## The pipeline contains 4 parts:
1.	Face recognition + cropping (for both the reference image and the test image).
2.	Encoding face images (for both the reference image and the test image).
3.	Distance calculation (between the reference image and the test image).
4.	Classification. (2 classes: Same person/different person)

## Cropping:
A class which perform cropping of an image, it uses a function detect_face() implemented in src/utils/utils.py
The output of the method cropp_image() is a cropped face.

## Encoding:
This step takes as input a cropped face (Output of Cropping) and return a vector of 512 elements using a pre-trained model on Inception ResNet v1 architecture trained on dataset VGGFace2
#### Architecture:
Inception ResNet v1:
Inspired by the performance of the ResNet, a hybrid inception module was proposed. There are two sub-versions of Inception ResNet, namely v1 and v2.
For more details visit : https://towardsdatascience.com/a-simple-guide-to-the-versions-of-the-inception-network-7fc52b863202
#### Training data:
VGGFace2 is a large-scale face recognition dataset. Images are downloaded from Google Image Search and have large variations in pose, age, illumination, ethnicity and profession.
The dataset contains 3.3M faces and 9000 classes.
For more details visit : https://www.robots.ox.ac.uk/~vgg/data/vgg_face2/

## Distance calculation:
Through Cropping + Encoding each image is transformed to a 512 vector, in this part of the pipeline we will compute the Euclidian distance between two vectors, this distance is high when the two images belong to two different persons, and is low if the two images belong to the same person.

## Classification:
Given 2 images (reference and test) we perform cropping + Encoding + DistanceCalculation.
Using a validation dataset of images of the same and different persons, we tuned a threshold. Thus if the distance between these 2 images is bigger than the threshold we classify the test images as (image of different persons) else we classify the image as (image of the same person).

## Project structure:

![Inkedstr project face verification_LI](https://user-images.githubusercontent.com/55580735/80012277-982c9c80-84bc-11ea-90d3-60af699a4418.jpg)

#### Files:
DataLoader: Load images from Data/input directory
Cropper: takes images loaded by DataLoader and cropp faces.
Encoder: takes cropped faces and return their embeddings (vector of 512 elements)
Distance Calculator: Calculate distance between the reference image and the test images.
Classifier: Classify test images to 2 classes (Same/Different person) the person is represented by the reference image.
DataWriter: Write a csv file labeling test images (Output of the pipeline).
Main: Calls the pipeline from data loading to data writing.

## Test pipeline Using a threshold of 1.1:
#### Input data:
Reference image:

<img width="563" alt="Capture" src="https://user-images.githubusercontent.com/55580735/80014599-0a52b080-84c0-11ea-8249-280ab81f6398.PNG">

Test images:

<img width="585" alt="test face" src="https://user-images.githubusercontent.com/55580735/80012483-eb065400-84bc-11ea-95e1-26993abeb605.png">

#### Output of the pipeline:

<img width="171" alt="output face ver" src="https://user-images.githubusercontent.com/55580735/80012501-efcb0800-84bc-11ea-9d81-022e086b369b.png">
