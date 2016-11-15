/////////////////////////////////////////////////////////////////////////////
//
// COMS30121 - darts.cpp
//
/////////////////////////////////////////////////////////////////////////////

// header inclusion
// header inclusion
#include <stdio.h>
//#include "opencv2/objdetect/objdetect.hpp"
//#include "opencv2/opencv.hpp"
//#include "opencv2/core/core.hpp"
//#include "opencv2/highgui/highgui.hpp"
//#include "opencv2/imgproc/imgproc.hpp"
#include <opencv2/opencv.hpp>
#include <iostream>
#include <stdio.h>
//#include "face_gt.h"
#include "darts_gt.h"
using namespace std;
using namespace cv;

/** Function Headers */
void detect( Mat& frame, vector<Rect>& output );
void drawRects(Mat& frame, Mat& output, vector<Rect> v);
double rectIntersection(Rect A, Rect B);
double fscore(vector<Rect> ground, vector<Rect> detected);

/** Global variables */
CascadeClassifier cascade;

/** @function main */
int main( int argc, const char** argv )
{
  // 1. Read Input Image
	Mat frame = imread(argv[1], CV_LOAD_IMAGE_COLOR);
	string fullname(argv[1]);
	int index = findIndexOfFile(fullname);
	//string noext = fullname.substr(0, fullname.find_last_of("."));

  vector<Rect> output;
	Mat frame_output;

	// 2. Load the Strong Classifier in a structure called `Cascade'
	if( !cascade.load( cascade_name ) ){ printf("--(!)Error loading\n"); return -1; };

	// 3. Detect dartboards
	detect( frame, output );

	// 4. Display result
	drawRects(frame, frame_output, output);


	// 5. Save Result Image
	//imwrite( noext+"_detected.jpg", frame );
	imwrite( "detected.jpg", frame_output );

	// 6. Measure performance if ground truth is available
	if(index>=0){
		vector<Rect> gt(positives[index], positives[index] + sizes[index]);
		cout << "*****F1 SCORE*****" << endl;
		cout << fscore(gt, output) << endl;
		cout << "******************" << endl;
		//Mat testRect;
		//drawRects(frame, testRect, gt);
		//imshow("Test Rects", testRect);
		//waitKey(0);
	}
	else{
		cout << "*****GROUND TRUTH NOT AVAILABLE******" << endl;
	}

	return 0;
}

/** @function detectAndDisplay */
void detect( Mat& frame, vector<Rect>& output )
{
	Mat frame_gray;
	//Mat grad_x, grad_y;
	//Mat abs_grad_x, abs_grad_y;
	//Mat grad;
	//string window_name = "Sobel test";

	// 1. Prepare Image by turning it into Grayscale and normalising lighting
	cvtColor( frame, frame_gray, CV_BGR2GRAY );
	equalizeHist( frame_gray, frame_gray );

	// 2. Perform Viola-Jones Object Detection
	cascade.detectMultiScale( frame_gray, output, 1.1, 1, 0|CV_HAAR_SCALE_IMAGE, Size(50, 50), Size(500,500) );

  //vector<Rect> gt(dart5gt, dart5gt + sizeof dart5gt/sizeof dart5gt[0]);
  //cout << fscore(gt, dartboards) << endl;

  // 3. Print number of dartboards found
	cout << output.size() << endl;

}

double rectIntersection(Rect A, Rect B){
	Point tlA = A.tl();
	Point brA = A.br();
	Point tlB = B.tl();
	Point brB = B.br();

  double inter_area = max(0, min(brA.x, brB.x) - max(tlA.x, tlB.x)) * max(0, min(brA.y,brB.y) - max(tlA.y, tlB.y));

	return inter_area / (A.area()+B.area()-inter_area);

}

void drawRects(Mat& frame, Mat& output, vector<Rect> v)
{

	output = frame.clone();

	for(unsigned int i = 0; i < v.size(); i++ )
	{
		rectangle(output, Point(v[i].x, v[i].y), Point(v[i].x + v[i].width, v[i].y + v[i].height), Scalar( 0, 255, 0 ), 2);
	}

}

double fscore(vector<Rect> ground, vector<Rect> detected)
{
	int TP = 0;
	double thresh = 0.4;
	vector<Rect>::iterator itG = ground.begin();
	while(itG!=ground.end()){
		bool matched = false;
		vector<Rect>::iterator itD = detected.begin();
		while(itD!=detected.end()){
			double interScore = rectIntersection(*itG, *itD);
			if(interScore > thresh){
				TP++;
				detected.erase(itD);
				itG = ground.erase(itG);
				matched = true;
				break;
			}
			else{
				++itD;
			}
		}
		if(!matched) ++itG;
	}

  int FP = detected.size();
	int FN = ground.size();

  if(TP+FN+FP == 0)
		return 1;
	return (double)2*TP / (double)(2*TP+FN+FP);
}
