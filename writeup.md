#**Traffic Sign Recognition** 

##Writeup Template

###You can use this file as a template for your writeup if you want to submit it as a markdown file, but feel free to use some other method and submit a pdf if you prefer.

---

**Build a Traffic Sign Recognition Project**

The goals / steps of this project are the following:
* Load the data set (see below for links to the project data set)
* Explore, summarize and visualize the data set
* Design, train and test a model architecture
* Use the model to make predictions on new images
* Analyze the softmax probabilities of the new images
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./examples/visualization.jpg "Visualization"
[image2]: ./examples/grayscale.jpg "Grayscaling"
[image3]: ./examples/random_noise.jpg "Random Noise"
[image4]: ./Images/00007.ppm "Traffic Sign 1"
[image5]: ./Images/00020.ppm "Traffic Sign 2"
[image6]: ./Images/07384.ppm "Traffic Sign 3"
[image7]: ./Images/07388.ppm "Traffic Sign 4"
[image8]: ./Images/07392.ppm "Traffic Sign 5"
[image9]: ./Images/exploration.png "Exploration"

[new_img1]: ./Images/new_img1.png "Traffic Sign 1"
[new_img2]: ./Images/new_img2.png "Traffic Sign 2"
[new_img3]: ./Images/new_img3.png "Traffic Sign 3"
[new_img4]: ./Images/new_img4.png "Traffic Sign 4"
[new_img5]: ./Images/new_img5.png "Traffic Sign 5"

## Rubric Points
###Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/481/view) individually and describe how I addressed each point in my implementation.  

---
###Writeup / README

####1. Provide a Writeup / README that includes all the rubric points and how you addressed each one. You can submit your writeup as markdown or pdf. You can use this template as a guide for writing the report. The submission includes the project code.

You're reading it! and here is a link to my [project code](https://github.com/chenguoming1/CarND-Traffic-Sign-Classifier-Project/blob/master/Traffic_Sign_Classifier.ipynb)

###Data Set Summary & Exploration

####1. Provide a basic summary of the data set. In the code, the analysis should be done using python, numpy and/or pandas methods rather than hardcoding results manually.

I used the pandas library to calculate summary statistics of the traffic
signs data set:

* The size of training set is 34799
* The size of the validation set is 4410
* The size of test set is 12630
* The shape of a traffic sign image is (32, 32)
* The number of unique classes/labels in the data set is 43

####2. Include an exploratory visualization of the dataset.

Data Exploration, according to the image, some of the front signs got more data then the rare, so there may overfit to the front signs.

![alt text][image9]

###Design and Test a Model Architecture

####1. Describe how you preprocessed the image data. What techniques were chosen and why did you choose these techniques? Consider including images showing the output of each preprocessing technique. Pre-processing refers to techniques such as converting to grayscale, normalization, etc. (OPTIONAL: As described in the "Stand Out Suggestions" part of the rubric, if you generated additional data for training, describe why you decided to generate additional data, how you generated the data, and provide example images of the additional data. Then describe the characteristics of the augmented training set like number of images in the set, number of images for each class, etc.)

Preprocessing Steps:
1) convert to gray scale, to make the performance faster
2) feature scaling (input - 128)/128 to get the input value within managable range.



####2. Describe what your final model architecture looks like including model type, layers, layer sizes, connectivity, etc.) Consider including a diagram and/or table describing the final model.

I am using Lenet model, the following is the detailed structure of the network.

Input layer = 32x32x1 

Convolution layer = 5x5 kernel, stride=1, padding=VALID, output=28x28x6, activation=relu

Avg pooling layer = 2x2 kernel, stride=2, padding=VALID, output=14x14x6

Convolution layer = 5x5 kernel, stride=1, padding=VALID, output=10x10x16, activation=relu

Max pooling lyer  = 2x2 kernel, stride=2, padding=VALID, output=5x5x16

Fully connected layer = 120 nodes, activation=relu

Fully connected layer = 84 nodes, activation=relu

Output layer = 43

learning rate = 0.005

classification function = softmax

optimizer = adam


####3. Describe how you trained your model. The discussion can include the type of optimizer, the batch size, number of epochs and any hyperparameters such as learning rate.

optimizer: adam 
batch size: 128
number of epoches: 10
learning rate: 0.005

####4. Describe the approach taken for finding a solution and getting the validation set accuracy to be at least 0.93. Include in the discussion the results on the training, validation and test sets and where in the code these were calculated. Your approach may have been an iterative process, in which case, outline the steps you took to get to the final solution and why you chose those steps. Perhaps your solution involved an already well known implementation or architecture. In this case, discuss why you think the architecture is suitable for the current problem.

Iterative approach is usually take very long time, for me it is better to use batch approach.

Lenet CNN's convolution layers serve as feature deduction and the fully connected layer serve as the actual training.

and it's easy to train thefore LeNet is suitable for current problem.

I have tried by adding more layers and more hidden units, the validation result seem better but when it come to test set the result is very porr, so it may be overfitting.
 
I have no idea about choosing parameter, tried adjusting the learning rates and find that learning rate 0.001 to 0.007 seem good and the rest are getting very poor result.

So i just tried to train with these numbers and sometimes it gets to 0.93 in validation set but sometimes doesn't so I tried to run multiple times and when it get to 0.93 i stopped the training and saved the model.

###Test a Model on New Images

####1. Choose five German traffic signs found on the web and provide them in the report. For each image, discuss what quality or qualities might be difficult to classify.

German traffic signs from the web:

![alt text][new_img1] ![alt text][new_img2] ![alt text][new_img3] 
![alt text][new_img4] ![alt text][new_img5]

Can only choose small and similar sized image since the larger images performance are very poor since the sign is very small and texture is big.

####2. Discuss the model's predictions on these new traffic signs and compare the results to predicting on the test set. At a minimum, discuss what the predictions were, the accuracy on these new predictions, and compare the accuracy to the accuracy on the test set (OPTIONAL: Discuss the results in more detail as described in the "Stand Out Suggestions" part of the rubric).

Here are the results of the prediction:

| Image			        |     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| Priority road      		| Priority road   									| 
| Speed limit (70km/h)     			| Speed limit (70km/h) 										|
| Yield					| Yield											|
| End of speed limit (80km/h)	      		| End of speed limit (80km/h)					 				|
| No vehicles			| No vehicles      							|


The model was able to correctly guess 5 of the 5 traffic signs, which gives an accuracy of 100%. 

The accuracy here is good because i choose the similar sized images, if I choose bigger images the performance worsen, so should try to add gittered images and train it to get better performance


####3. Describe how certain the model is when predicting on each of the five new images by looking at the softmax probabilities for each prediction. Provide the top 5 softmax probabilities for each image along with the sign type of each probability. (OPTIONAL: as described in the "Stand Out Suggestions" part of the rubric, visualizations can also be provided such as bar charts)

Currently the logic is working 100% accuracy for the images with similar size and shape,
it has very poor result in large images, i still need to explore more to discuss about it, for now no idea yet for discuss further.

The top five soft max probabilities:


****top five probabilities for image 1  ****

| Image|Probabilities|
|:---------------------:|:---------------------------------------------:|
0 5
| Priority road | 99.9989748001 %|
| Speed limit (30km/h) | 0.000584681902183 %|
| Speed limit (50km/h) | 0.00023453758331 %|
| Roundabout mandatory | 0.00010265474657 %|
| Speed limit (70km/h) | 8.04954595424e-05 %|

****top five probabilities for image 2  ****

| Image|Probabilities|
|:---------------------:|:---------------------------------------------:|
1 5
| Speed limit (70km/h) | 95.9743022919 %|
| Speed limit (120km/h) | 1.86452642083 %|
| Speed limit (30km/h) | 1.54375638813 %|
| Speed limit (100km/h) | 0.534782465547 %|
| Speed limit (20km/h) | 0.0367628934328 %|

****top five probabilities for image 3  ****

| Image|Probabilities|
|:---------------------:|:---------------------------------------------:|
2 5
| Yield | 99.9993801117 %|
| Priority road | 0.000345543253388 %|
| No passing | 0.000267312702817 %|
| Ahead only | 1.25945049945e-05 %|
| Children crossing | 1.4344546384e-08 %|

****top five probabilities for image 4  ****

| Image|Probabilities|
|:---------------------:|:---------------------------------------------:|
3 5
| End of speed limit (80km/h) | 99.7189939022 %|
| Speed limit (30km/h) | 0.122778338846 %|
| End of no passing by vehicles over 3.5 metric tons | 0.10633567581 %|
| End of all speed and passing limits | 0.0490245060064 %|
| Speed limit (60km/h) | 0.00168548649526 %|

****top five probabilities for image 5  ****

| Image|Probabilities|
|:---------------------:|:---------------------------------------------:|
4 5
| No vehicles | 33.6468070745 %|
| Speed limit (80km/h) | 19.6046367288 %|
| Speed limit (120km/h) | 16.5354669094 %|
| Speed limit (50km/h) | 15.4473975301 %|
| Priority road | 6.20617754757 %|




