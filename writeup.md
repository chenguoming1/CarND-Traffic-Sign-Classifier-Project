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

| Image                 |Probabilities                                  |
|:---------------------:|:---------------------------------------------:|
| Priority road        | 99.9948263168 %|
| Speed limit (50km/h) | 0.00313042182825 %|
| Roundabout mandatory | 0.000792487026047 %|
| Speed limit (30km/h) | 0.000609912967775 %|
| Speed limit (70km/h) | 0.000334634955834 %|

****top five probabilities for image 2  ****

| Image                 |Probabilities                                  |
|:---------------------:|:---------------------------------------------:|
| Speed limit (70km/h) | 97.9661941528 %|
| Speed limit (30km/h) | 1.64432302117 %|
| Speed limit (120km/h) | 0.231515383348 %|
| Speed limit (50km/h) | 0.070565653732 %|
| Speed limit (20km/h) | 0.0263143505435 %|

****top five probabilities for image 3  ****

| Image                 |Probabilities                                  |
|:---------------------:|:---------------------------------------------:|
| Yield | 100.0 %|
| Turn left ahead | 1.54500284094e-10 %|
| Priority road | 3.58879267449e-11 %|
| Keep right | 4.48695171729e-12 %|
| Children crossing | 8.79885458927e-13 %|

****top five probabilities for image 4  ****

| Image                 |Probabilities                                  |
|:---------------------:|:---------------------------------------------:|
| End of speed limit (80km/h) | 99.695122242 %|
| End of no passing by vehicles over 3.5 metric tons | 0.1607531216 %|
| End of all speed and passing limits | 0.106064439751 %|
| Speed limit (30km/h) | 0.0311509356834 %|
| End of no passing | 0.00392342808482 %|

****top five probabilities for image 5  ****

| Image                 |Probabilities                                  |
|:---------------------:|:---------------------------------------------:|
| No vehicles | 93.2538807392 %|
| Speed limit (50km/h) | 2.16651428491 %|
| Priority road | 2.01165750623 %|
| Speed limit (120km/h) | 0.80029508099 %|
| Speed limit (70km/h) | 0.52204108797 %|

[[  9.99948263e-01   3.13042183e-05   7.92487026e-06   6.09912968e-06
    3.34634956e-06]
 [  9.79661942e-01   1.64432302e-02   2.31515383e-03   7.05656537e-04
    2.63143505e-04]
 [  1.00000000e+00   1.54500284e-12   3.58879267e-13   4.48695172e-14
    8.79885459e-15]
 [  9.96951222e-01   1.60753122e-03   1.06064440e-03   3.11509357e-04
    3.92342808e-05]
 [  9.32538807e-01   2.16651428e-02   2.01165751e-02   8.00295081e-03
    5.22041088e-03]]
[[12  2 40  1  4]
 [ 4  1  8  2  0]
 [13 34 12 38 28]
 [ 6 42 32  1 41]
 [15  2 12  8  4]]




