Object Detection
==================

Object detection consists of two sub-tasks: 
- localization, which is determining the location of an object in an image, and 
- classification, which is assigning a class to that object

[Find Most popular metrics used to evaluate object detection algorithms](https://github.com/rafaelpadilla/Object-Detection-Metrics).


Object Detection Competitions
==============================
1) COCO
2) Pascal VOC (Visual Object Classifier) 07/12
3) ImageNet

Bounding Box
=============

Create a rectangle. Take two coordinates or center coordinates with width/height of the box to construct bouding box. Either considereding top-left corner and bottom-right corner coordinates or consideirng center point and with width/height of the box. Loss function is being calculated between predicted coordinates and actual coordinates of these.

Bounding Box Regression
=======================

Bounding Box regression is a metric for measuring how well predicted bounding boxes captures objects. Determine quantities. 

This is done by calculating the mean-squared error (MSE) between the coordinates of predicted bounding boxes and the ground truth bounding boxes.

Example:

Actual:
(x, y) = (350, 350)
l = 50
h = 50

Predicted:
(x', y') = (310, 310)
l' = 10
h' = 10

Loss Function to use:
1) L1 loss 
	- L1 loss is also known as mean absolute loss or mean absolute error.
	- MAE = (1/n) * Σ|yᵢ - ȳ|
2) L2 loss 
	- The Mean Square Error(MSE) or L2 loss is a loss function that quantifies the magnitude of the error between a machine learning algorithm prediction and an actual output by taking the average of the squared difference between the predictions and the target values.
	- MSE = (1/n) * Σ(yᵢ - ȳ)²
3) Smooth-L1 loss
	- This version of loss depends on the value based on the value of the beta threshold. When the value is less than threshold It’s less sensitive to outliers than the mean squared error loss. But at the same time when the value is greater, it acts the same as the L1 loss which makes it more sensitive to the outliers.

Intersection Over Union (IOU)
=============================

Intersection over Union (IoU), also known as the Jaccard index, is the most popular evaluation metric for tasks such as segmentation, object detection and tracking. 

IOU evaluates the overlap between two bounding boxes. It requires a ground truth bounding box and a predicted bounding box. By applying IOU we can tell if a detection is valid (True +ve) or not (False +ve).

IOU = (area of overlap)/(area of union)

Precision and Recall
======================

Precision:
- Precision is the ability of a model to identify only the relevant objects. It is the percentage of correct +ve predictions and is given by the following eq.
- Precision = TP/(TP + FP) = TP/(all detection)
- TP -> True Positive: A correct detection
- FP -> False Positive: A wrong detection
- FN -> False Negative: A ground truth not detected
- TN -> True Negative: Not detected correctly  

Recall: 
- It is the ability of a model to find all the relevant cases (all ground truth bounding boxes). It is the percentage of true positive detected among all relevant ground truths and is given by:
- Recall = TP/(TP + FN) = TP/(all ground truths)

Average Precision (AP)
========================

The average precision (AP) is a way to summarize the precision-recall curve into a single value representing the average of all precisions. AP summarizes a precision-recall curve as the weighted mean of precisions achieved at each threshold, with the increase in recall from the previous threshold used as the weight:

- AP = Σ(Rn - Rn-1)*Pn
- Pn -> Precision at the n-th threshold
- Rn -> Recall at the n-th threshold

Mean Average Precision (mAP)
=============================


Object Detection Architectures
=================================
1) OD Family (3 family)
	- RCNN Family (Slower and more accurate)
		- RCNN
		- Fast RCNN
		- Faster RCNN
		- Mask RCNN
	- Single Shot Detector (SSD) Family (faster and less accurate)
		- YOLO
		- SSD
	- CenterNet Family (slow and better accuracy)
		- CENTERNET

RCNN
========

Paper: https://arxiv.org/pdf/1311.2524.pdf

R-CNN -> Region Proposals + CNN

Steps of RCNN/Architecture of RCNN:
1) Takes an input image
2) Extract around 2000 bottom-up region proposal using selective serach method
3) Warped regions
4) Computes features for each proposal using a large CNN (AlexNet)
5) 4.a: Classifies each region using class-specific linear SVMs (from CNN features).
6) 4.b: Bounding box prediction using linear regression (from CNN features) 

Stages of RCNN:
1) The first module generates category-independent region proposals.
2) The second module is a large convolutional neural netork that extracts a fixed-length feature vector from each region.
3) The third module is a set of class-specific linear SVMs.


Methods of Region Proposals:
1) Objectness
2) Selective search (selected for RCNN)
3) Category-independent object proposals
4) Constrained parametric min-cuts (CPMC)
5) Multi-scale combinational grouping
6) Ciresan et al.

Feature Extraction:
1) ~2000 region proposal extracted from selective search method
2) Extracted 4096-dimensional feature vector from each region proposal using Caffe implementation of the CNN described by Krizhevsky et al.
3) Features are computed by forward propagating a mean-subtracted 227 × 227 RGB image through five convolutional layers and two fully connected layers.


Fast RCNN
===============

Paper: https://arxiv.org/pdf/1504.08083.pdf

Source Code: https://github.com/rbgirshick/fast-rcnn

Architcture:
1) Input Image -> (Selective Search+Edge Box)
2) Input Image and (Selective Search+Edge Box) -> AlexNet -> Features -> RoI Pooling
-> FC -> FC
3) 2.a) Softmax (for classification object)
4) 2.b) Bounding Box Regression (L2 Loss and SmoothL1 Loss) for bounding box prediction of the object


Faster RCNN
=================

Paper: https://arxiv.org/pdf/1506.01497.pdf

Source Code: https://github.com/rbgirshick/py-faster-rcnn

Architcture:
1) Input Image -> VGG -> Features <-> RPN -> RoI Pooling -> FC -> FC -> a) Fine Tuning Softmax using Log Loss and b) Bounding Box Regression
2) RPN (Regional Proposal Network) -> 3x3 -> a) 1x1 for classification (FG/BG) and b) 1x1 for Region Bounding Box


YOLO
==============

Paper: https://arxiv.org/pdf/1506.02640.pdf

Website: https://pjreddie.com/darknet/yolo/

Architcture:
YOLO architecture is similar to GoogleNet. As illustrated below, it has overall 24 convolutional layers, four max-pooling layers, and two fully connected layers.

YOLO Flow:
1) Resizes the input image into 448x448 before going through the convolutional network.
2) A 1x1 convolution is first applied to reduce the number of channels, which is then followed by a 3x3 convolution to generate a cuboidal output.
3) The activation function under the hood is ReLU, except for the final layer, which uses a linear activation function.
4) Some additional techniques, such as batch normalization and dropout, respectively regularize the model and prevent it from overfitting.

The algorithm works based on the following four approaches:
1) Residual blocks
This first step starts by dividing the original image (A) into NxN grid cells of equal shape, where N in our case is 4 shown on the image on the right. Each cell in the grid is responsible for localizing and predicting the class of the object that it covers, along with the probability/confidence value. 

2) Bounding box regression
The next step is to determine the bounding boxes which correspond to rectangles highlighting all the objects in the image. We can have as many bounding boxes as there are objects within a given image. 
YOLO determines the attributes of these bounding boxes using a single regression module in the following format, where Y is the final vector representation for each bounding box. 
Y = [pc, bx, by, bh, bw, c1, c2]

3) Intersection Over Unions or IOU for short
4) Non-Maximum Suppression. 

Steps to follow:
1) Input image
2) Breaking an image into a grid
3) Each section of the grid is classified and localized
4) Predicts where to place bounding boxes based on regression-based algorithms

Requirements:
1) Yolov5 Repository
2) Python 3.8 or later
3) PyTorch (It is the most popular machine learning frameworks to define models, and perform training)
4) CUDA (It is NVIDIA's parallel computing platform for their GPUs)

Step-1: Install YOLOv5. Get YOLOv5 from here: 
- i) manual download https://github.com/ultralytics/yolov5
- ii) git clone https://github.com/ultralytics/yolov5.git

Step-2: Install Python. get this from here:
- https://www.python.org/downloads/

Step-3: Install CUDA. Get this from here:
- https://developer.nvidia.com/cuda-downloads

Step-4: Install PyTorch as a Python module. Get the correct pip install version of this by selecting the preferences available at here: https://pytorch.org/get-started/locally/

Step-5: Install additional modules for YOLOv5 available in requirements.txt inside of this folder
- pip install -r requirements.txt
- If visual studio error occure then get it from here to install:
    - https://visualstudio.microsoft.com/downloads/
- After that install PyCOCOTools:
    - pip install git+https://github.com/philferriere/cocoapi.git#subdirectory=PythonAPI
- Then further run this:
    - pip install -r requirements.txt
    - This is just for good measure.

Further details can be found from this links:
1) [YOLO Object Detection Explained](https://www.datacamp.com/blog/yolo-object-detection-explained)
2) [What Is YOLO Algorithm?](https://www.baeldung.com/cs/yolo-algorithm)
3) [YOLOv5 Object Detection on Windows (Step-By-Step Tutorial)](https://wandb.ai/onlineinference/YOLO/reports/YOLOv5-Object-Detection-on-Windows-Step-By-Step-Tutorial---VmlldzoxMDQwNzk4)


Practical
==============
1) Inference using pretrained YOLOv5 model with Google Colab
- Open Google Colab/Jupyter Notebook
- Cloning Github Repository (! git clone https://github.com/ultralytics/yolov5)
- Installling dependencies (! pip install -r requirements.txt)
- Dependency Checks

```
import torch

from IPython.display import Image, clear_output  # to display images

clear_output()

print(
    f"Setup complete. Using torch \
    {torch.__version__} \
    ( \
        {torch.cuda.get_device_properties(0).name if torch.cuda.is_available() else 'CPU'} \
    )" \
)

# inferencing with YoloV5

# inference with default model weights

! python detect.py --img 640 --conf 0.4 --source /full/path/of/the/test-image-folder

# inference with different model weights (see 'Pretrained Checkpoints' for model weights and pixel size/resolution at: https://github.com/ultralytics/yolov5)

! python detect.py --img 640 --conf 0.4 --weights yolov5m.pt --source /full/path/of/the/test image folder
```

2) Inference using pretrained YOLOv5 model on Local System
```
## Using Webcamp Data
#=====================
# in cmd terminal, cd to YOLOv5 directory: > cd tolov5
# type this line of code: python detect.py --source 0
# Note: If there is multiple webcams or applications that spoof them we may need to change this number. If there is only one, then source will be 0 as in the example code.
# To stop taking this simply click inside the open window.

## Using Image Data
#==================
# If you want to detect objects in a pre-existing image, simply place the image in a folder within the YOLO directory. 
# Opening up the YOLO Console again, we enter:
# python detect.py --source images/google_image.jpg
# Go to the location it's stored and see the new image with resultant bounding boxes

## Using YouTube Video Data
# Install few more modules to create bounding boxes or just detect objects on a YouTube video
# pip install pafy
# pip install youtube_dl
# Get the youtube video id from video link (video id of this https://www.youtube.com/watch?v=jNQXAC9IVRw link is 'jNQXAC9IVRw')
# Construct source link of the video id: https://youtu.be/jNQXAC9IVRw
# Now in cmd terminal type this line to run: python detect.py --source https://youtu.be/jNQXAC9IVRw
# press ctrl+C in terminal to stop
# tested with this: python detect.py --source https://youtu.be/AmarqUQzqZg

```

3) Custom training and inference of YOLO model
- Get new images
- Download LabelImg annotation tool from here: https://github.com/HumanSignal/labelImg
- Do annotation
- <in progress>

SSD
===========

Paper: https://arxiv.org/pdf/1512.02325.pdf

Source Code: https://github.com/weiliu89/caffe/tree/ssd

Architcture:
1) A Backbone Model -> A pre-trained image classification network serves as the backbone model's feature extractor. Usually, this is a network similar to ResNet that was trained on ImageNet after the last fully connected classification layer was eliminated. 
2) SSD Head -> The outputs are interpreted as bounding boxes and classes of objects in the spatial location of the final layer activations. The SSD head is simply one or more convolutional layers added to this backbone.
3) SSD uses a grid to split the image, with each grid cell in charge of identifying objects in that area.
4) In SSD, multiple anchor/prior boxes can be assigned to each grid cell. Each of these pre-defined anchor boxes is in charge of a specific size and shape inside a grid cell. 
5) During training, SSD employs a matching phase to align the bounding boxes of each ground truth object in an image with the relevant anchor box. 
6) Predicting the class and location of an object is essentially the responsibility of the anchor box that overlaps with it the most.
7) This can be accommodated by the anchor boxes' pre-defined aspect ratios thanks to the SSD architecture. The various aspect ratios of the anchor boxes associated with each grid cell at each zoom/scale level can be specified using the ratios parameter.
8) The amount that the anchor boxes need to be scaled up or down in relation to each grid cell is specified by the zooms parameter.

Further details can be found from this links:
1) [How single-shot detector (SSD) works?](https://developers.arcgis.com/python/guide/how-ssd-works/)
2) [Single Shot Detectors (SSDs)](https://www.baeldung.com/cs/ssd)

SSD Flow:
1) Process the image
2) Pass the image through model
3) Apply anchor boxes
4) Run the classifier
5) Non-maximal suppression
6) Return the bounding boxes and class probabilities
7) Done
