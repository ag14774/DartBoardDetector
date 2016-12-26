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

