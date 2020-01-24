# **Traffic Sign Recognition** 

## Writeup


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

[image1]: ./img_show/dataset.png "Visualization"
[image2]: ./img_show/grayset.png "Grayscaling"
[image3]: ./examples/random_noise.jpg "Random Noise"
[image4]: ./img_show/img5.png "Traffic Sign 1"

## Rubric Points
### Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/481/view) individually and describe how I addressed each point in my implementation.  

---

### Data Set Summary & Exploration

#### 1. Provide a basic summary of the data set. In the code, the analysis should be done using python, numpy and/or pandas methods rather than hardcoding results manually.

I used the pandas library to calculate summary statistics of the traffic
signs data set:

* Number of training examples = 34799
* Number of testing examples = 12630
* Image data shape = (32, 32, 3)
* Number of classes = 43

#### 2. Include an exploratory visualization of the dataset.

Here is an exploratory visualization of the data set. 

![alt text][image1]

### Design and Test a Model Architecture

#### 1. Describe how you preprocessed the image data. What techniques were chosen and why did you choose these techniques? Consider including images showing the output of each preprocessing technique. Pre-processing refers to techniques such as converting to grayscale, normalization, etc. (OPTIONAL: As described in the "Stand Out Suggestions" part of the rubric, if you generated additional data for training, describe why you decided to generate additional data, how you generated the data, and provide example images of the additional data. Then describe the characteristics of the augmented training set like number of images in the set, number of images for each class, etc.)

As a first step, I decided to convert the images to grayscale because color does not affect judgment.

Here is an example of a traffic sign image before and after grayscaling.

![alt text][image2]

As a last step, I normalized the image data because normalization reduces the complexity of the problem the network is trying to solve. This can potentially increase the accuracy of the model and speed up the training


#### 2. Describe what your final model architecture looks like including model type, layers, layer sizes, connectivity, etc.) Consider including a diagram and/or table describing the final model.

My final model consisted of the following layers:

| Layer         		|     Description	        					| 
|:---------------------:|:---------------------------------------------:| 
| Input         		| 32x32x1 RGB image   							| 
| Convolution 5x5     	| 1x1 stride, valid padding, outputs 28x28x6 	|
| RELU					|												|
| Max pooling	      	| 2x2 stride,  outputs 14x14x6				|
| Convolution 5x5	    | 1x1 stride, valid padding, outputs 10x10x16      									|
| RELU		|    									|
| Max pooling					| 2x2 stride,  outputs 5x5x16.    									|
|		Fully Connected		|							Input = 400. Output = 120					|
|			RELU			|												|
|			 Fully Connected			|			 Input = 120. Output = 84									|
|				RELU		|												|
|			Fully Connected		|			Input = 84. Output = 43									|
|						|												|
 


#### 3. Describe how you trained your model. The discussion can include the type of optimizer, the batch size, number of epochs and any hyperparameters such as learning rate.

To train the model, I used 

| Parameter         		|     Value	        					| 
|:---------------------:|:--------------------------------:| 
| EPOCHS        |   30          |
|   BATCH_SIZE|  100       | 
|  Training rate | 0.001          | 
| Optimizer  | Adam Optimizer          | 
|						|												|

#### 4. Describe the approach taken for finding a solution and getting the validation set accuracy to be at least 0.93. Include in the discussion the results on the training, validation and test sets and where in the code these were calculated. Your approach may have been an iterative process, in which case, outline the steps you took to get to the final solution and why you chose those steps. Perhaps your solution involved an already well known implementation or architecture. In this case, discuss why you think the architecture is suitable for the current problem.

My final model results were:

* validation set accuracy of 96.8% 
* test set accuracy of 93.8%


If a well known architecture was chosen:
* What architecture was chosen: LeNet
* How does the final model's accuracy on the training, validation and test set provide evidence that the model is working well? It worked well
 

### Test a Model on New Images

#### 1. Choose five German traffic signs found on the web and provide them in the report. For each image, discuss what quality or qualities might be difficult to classify.

Here are five German traffic signs that I found on the web:

![alt text][image4] 


#### 2. Discuss the model's predictions on these new traffic signs and compare the results to predicting on the test set. At a minimum, discuss what the predictions were, the accuracy on these new predictions, and compare the accuracy to the accuracy on the test set (OPTIONAL: Discuss the results in more detail as described in the "Stand Out Suggestions" part of the rubric).

Here are the results of the prediction:

| Image			        |     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| Turn right ahead     		| Turn right ahead									| 
| Keep right     			| Keep right										|
| Vehicles over 3.5 metric tons prohibited					| Vehicles over 3.5 metric tons prohibited											|
| Speed limit (30km/h)      		| Speed limit (30km/h)					 				|
| Right-of-way at the next intersection		| Right-of-way at the next intersection						|


The model was able to correctly guess 5 of the 5 traffic signs, which gives an accuracy of 100%. 

#### 3. Describe how certain the model is when predicting on each of the five new images by looking at the softmax probabilities for each prediction. Provide the top 3 softmax probabilities for each image along with the sign type of each probability. (OPTIONAL: as described in the "Stand Out Suggestions" part of the rubric, visualizations can also be provided such as bar charts)



##### image: Turn right ahead  
| Label         | Probability         	|     Prediction	        					|             
|:---------------:|:-----------------:|:---------------------------------------------:| 
|      33       | 1.0000000e+00         			| Turn right ahead  									| 
|       12      | 2.4246347e-08  				| Priority road										|
|        15     | 1.7683403e-08				| No vehicles									|


##### image: Keep right  
| Label         | Probability         	|     Prediction	        					|             
|:---------------:|:-----------------:|:---------------------------------------------:| 
|      38       | 1.0000000e+00         			| Keep right									| 
|       13      | 1.0395547e-36				| Yield										|
|        34     | 2.9915370e-37			|  Turn left ahead									|



##### image: Vehicles over 3.5 metric tons prohibited	
| Label         | Probability         	|     Prediction	        					|             
|:---------------:|:-----------------:|:---------------------------------------------:| 
|      16      | 1.0000000e+00         			| Vehicles over 3.5 metric tons prohibited   									| 
|       9      | 8.4887277e-14 				| No passing										|
|        42     | 3.2397219e-15			| End of no passing by vehicles over 3.5 metric tons										|



##### image: Speed limit (30km/h)  
| Label         | Probability         	|     Prediction	        					|             
|:---------------:|:-----------------:|:---------------------------------------------:| 
|      1       | 1.0000000e+00         			| Speed limit (30km/h)  									| 
|       2      | 2.9334696e-10				| Speed limit (50km/h)									|
|        0     | 2.0192398e-14			| Speed limit (20km/h)										|



##### image: Right-of-way at the next intersection
| Label         | Probability         	|     Prediction	        					|             
|:---------------:|:-----------------:|:---------------------------------------------:| 
|      11       | 9.9999964e-01    			| Right-of-way at the next intersection									| 
|       30      | 3.4079059e-07 				| Beware of ice/snow										|
|        27     | 2.4105409e-09		| Pedestrians										|






