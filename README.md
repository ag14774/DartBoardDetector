#Task Overview: Detecting Dart Boards

##Introduction
Being able to detect and locate instances of an object class in images is important for many computer vision applicationsas well as an ongoing area of research. This assignment requires you 1) to experiment with the Viola-Jones object detection framework as discussed in the lectures and provided by OpenCV, and 2) combine it with other detection approaches to improve its efficacy. Your approach is to be tested and evaluated on a small image set that depicts aspects of a popular sport and past time – darts.

##First
There is an introductory part which helps you to understand how the Viola-Jones detector can be used in OpenCV and lets you experience how it operates in practice. In particular, you will experiment with an off-the-shelf face detector.
Resources:
-https://en.wikipedia.org/wiki/Cascading_classifiers#Cascade_Training
-http://www.uta.fi/sis/tie/tl/index/Rates.pdf
-https://en.wikipedia.org/wiki/Sensitivity_and_specificity
-https://en.wikipedia.org/wiki/Precision_and_recall
-http://link.springer.com/referenceworkentry/10.1007%2F978-0-387-39940-9_483
-https://en.wikipedia.org/wiki/F1_score

##Second
You will build your own object detector by training the Viola-Jones framework on a customized object class, that is 'dartboards'. You will then test the detector’s performance and interpret your results on an image set.

##Third
You will combine your detector with a Hough transform for component configurations of the dartboard shape in order to improve detections. A Hough transform (e.g. for concentric lines, circles, ellipses) should be used, combined and evaluated together with the Viola Jones detector.

##Fourth
In order to be able to reach marks above the first class boundary, we ask you to research, understand and use in OpenCV one other appropriate vision approach to further improve the detection efficacy and/or efficiency.

#Important Notes from OpenCV docs
##The correct way to scan an image
The ".at" function should be used when access to arbitrary/random elements is required. It is the least efficient way of looping through a whole image. The correct way is described here:
http://docs.opencv.org/2.4/doc/tutorials/core/how_to_scan_images/how_to_scan_images.html#the-efficient-way
or
http://docs.opencv.org/2.4/doc/tutorials/core/how_to_scan_images/how_to_scan_images.html#the-iterator-safe-method

##Typecasting
Use saturated cast when converting from one primitive type to another to take care of the max and min values:
http://docs.opencv.org/2.4/modules/core/doc/utility_and_system_functions_and_macros.html#saturate-cast
e.g. uchar a = saturate_cast<uchar>(-100);
This will set the unsigned char to 0 because -100 is out of range. If normal typecast were to be used -100, its binary representation would just be interpreted as unsigned.

##Changing the contrast and brightness
g(i,j) = a*f(i,j) + b
http://docs.opencv.org/3.1.0/d3/dc1/tutorial_basic_linear_transform.html

##Convolutions
Use filter2D function. filter2D just does correlation, so invert kernel first.
http://docs.opencv.org/3.1.0/d7/d37/tutorial_mat_mask_operations.html

F1 scores - darts = 2TP/(2TP+FN+FP)
            (TP) (FN)
dart0:  1/7  (1) (0)
dart1:  2/11 (1) (0)
dart2:  1/3  (2) (0)
dart3:  2/13 (1) (0)
dart4:  1/6  (1) (0)
dart5:  2/19 (1) (0)
dart6:  2/15 (1) (0)
dart7:  2/29 (1) (0)
dart8:  4/23 (2) (0)
dart9:  4/11 (2) (0)
dart10: 6/35 (3) (1)
dart11: 2/5  (2) (1)
dart12: 2/5  (1) (0)
dart13: 1/8  (1) (0)
dart14: 4/39 (2) (0)
dart15: 2/9  (1) (0)

-Initially with high sensitivity threshold. High TPR but also high FPR
-Each stage adds more features which discards FPs without decreasing TPs.
-This is like a constraint optimization problem, where a larger set of objects is initially detected, and we optimize by removing false positives.
-In each stage FPR decreases rapidly, hence extra stages could be helpful.
