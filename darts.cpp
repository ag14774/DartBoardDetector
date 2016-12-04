/////////////////////////////////////////////////////////////////////////////
//
// COMS30121 - darts.cpp
//
/////////////////////////////////////////////////////////////////////////////

// header inclusion
#include <stdio.h>
#include <algorithm>
#include <vector>
//#include "opencv2/objdetect/objdetect.hpp"
//#include "opencv2/opencv.hpp"
//#include "opencv2/core/core.hpp"
//#include "opencv2/highgui/highgui.hpp"
//#include "opencv2/imgproc/imgproc.hpp"
#include <opencv2/opencv.hpp>
#include <iostream>
#include <stdio.h>
#include <omp.h>
//#include "face_gt.h"
#include "darts_gt.h"

#define MAX_CONCENTRIC 20
//#define DEBUG
//#define GROUND_TRUTH
//#define ONLY_VIOLA_JONES
//#define NO_ELLIPSE_DETECTION

using namespace std;
using namespace cv;

const double pi = atan(1)*4;

typedef struct{
	int xc;
	int yc;
	int num;
	int rs[MAX_CONCENTRIC];
	int total_acc;
}ConcentricCircles;

/*
 * Structure that holds
 * information about an edge pixel
 */
typedef struct{
	int x; //x-coordinate
	int y; //y-coordinate
	double grad; //gradient of pixel (-pi to pi)
}EdgePointInfo;

/*
 * Structure to hold information
 * about an ellipse. Used by the
 * ellipse detector.
 */
typedef struct MyEllipse{
	MyEllipse(int nxc, int nyc, float nangle, int nmajor, int nminor, float naccum)
	{
		xc=nxc;
		yc=nyc;
		angle=nangle;
		major=nmajor;
		minor=nminor;
		accum=naccum;
	}
	int xc;      //x-coordinate
	int yc;      //y-coordinate
	float angle; //rotation of ellipse in radians
	int major;   //half-major of ellipse
	int minor;   //half-minor of ellipse
	float accum; //accumulator value from detection algorithm

}MyEllipse;

/** Function Headers */
void detect( Mat& frame, vector<Rect>& output );
void drawRects(Mat& frame, Mat& output, vector<Rect> v);
void HoughLinesFilter(const Mat& frame_gray, vector<Rect>& output);
void detectConcentric(vector<EdgePointInfo>& edgeList, Size imsize, int min_radius, int max_radius,
											int threshold, int resX, int resY, int resR, vector<ConcentricCircles>& output);
void show3Dhough(Mat& input);
void detectEllipse(vector<EdgePointInfo>& edgeList, Size imsize, vector<MyEllipse>& output, int threshold, int minMajor, int maxMajor);
void extractEdges(Mat& gray_input, vector<Rect>& v, vector<EdgePointInfo>& edgeList, int method, int edge_thresh, int kernel_size);
Point mergeCloseCenters(vector<Point>& candidate_centers, Size imsize);
double rectIntersection(Rect A, Rect B);
double fscore(vector<Rect> ground, vector<Rect> detected);

//******TODO*********
//1)Redo fscore and rectIntersection: Use intersection operator --> DONE
//2)Review detector flowchart
//3)Implement ellipse detector --> DONE - Needs tweaking
//4)Change step 7 of "detect" to choose the closest center instead of the first --> DONE
//5)Function to convert 3D hough space to 2D --> DONE
//6)Add gaussian blur to accumulator in ellipse detector --> DONE
//7)Coin flip for each pair of points instead of randomly permuting the points and picking the first X pairs(possible speedup)

/** Global variables */
CascadeClassifier cascade;

/** @function main */
int main( int argc, const char** argv )
{
	//Initialise the random number generator
	srand(time(0));

  // 1. Read Input Image
	Mat frame = imread(argv[1], CV_LOAD_IMAGE_COLOR);

	string fullname(argv[1]);
	// Used for loading ground truths
	int index = findIndexOfFile(fullname);
	//string noext = fullname.substr(0, fullname.find_last_of("."));

  vector<Rect> output;
	Mat frame_output;

	// 2. Load the Strong Classifier in a structure called `Cascade'
	if( !cascade.load( cascade_name ) ){ printf("--(!)Error loading\n"); return -1; };

	// 3. Detect dartboards
	detect( frame, output );
  cout<<"Number of dartboards found: "<<output.size()<<endl<<endl;

	// 4. Draw result
	drawRects(frame, frame_output, output);

	// 5. Save Result Image
	imwrite( "detected.jpg", frame_output );
	//imwrite( noext+"_detected.jpg", frame );

	// 6. Measure performance if ground truth is available
	if(index>=0){
		vector<Rect> gt(positives[index], positives[index] + sizes[index]); //load ground truth and convert to vector
		cout << "*****F1 SCORE*****" << endl;
		cout << fscore(gt, output) << endl;
		cout << "******************" << endl;
		#ifdef GROUND_TRUTH
		Mat testRect;
		drawRects(frame, testRect, gt);
		imshow("Test Rects", testRect);
		waitKey(0);
		#endif
	}
	else{
		cout << "*****GROUND TRUTH NOT AVAILABLE******" << endl;
	}

	return 0;
}

/* @function detect
 * Takes an image as input and outputs
 * an array of rectangles over the detected
 * dartboards. frame is not modified.
 */
void detect( Mat& frame, vector<Rect>& output )
{
	Mat frame_gray;										/* Grayscale version of image */
	Mat frame_gray_norm;							/* Grayscale version of image with normalised lighting */
	Mat frame_blur,frame_dst;
	vector<Rect> finalOut;						/* Temporary array of final dartboards */
	vector<Point> candidate_centers;  /* Candidate dartboard centers */


	// 1. Prepare Image by turning it into Grayscale
	cvtColor( frame, frame_gray, CV_BGR2GRAY );
	//Normalise lighting
  equalizeHist( frame_gray, frame_gray_norm );

	// 2. Perform Viola-Jones Object Detection
	cout << "**************Performing Viola-Jones**************" << endl;
	cascade.detectMultiScale( frame_gray_norm, output, 1.1, 1, 0|CV_HAAR_SCALE_IMAGE, Size(50, 50), Size(500,500) );
	cout << "Dartboards found using Viola-Jones: "<<output.size()<<endl;
	cout << "**************************************************" << endl<<endl;

  #ifdef ONLY_VIOLA_JONES
	return;
	#endif


// 3. Extract edges as a 1D array for ConcentricCircles
  cout << "****Preparing for concentric circle detection*****" << endl;
	vector<Rect> dummy;	//Pass an empty mask to extractEdges
  vector<EdgePointInfo> edges;
	//extractEdges applies no mask when the mask array is empty
	int method = 1; //Use Canny edge detector
	int edge_thresh = 50; //Threshold to be used in Canny
	int kernel_size = 5; //Blur before extracting edges with a 5x5 kernel
	//vector array "edges" holds information on where edge pixels were detected + gradient
	extractEdges(frame_gray, dummy, edges, method, edge_thresh, kernel_size);
	cout<<"Edge pixels found for circle detection: "<<edges.size()<<endl;
	cout << "**************************************************" << endl<<endl;


// 4. Detect concentric circles
  cout << "**********Detecting concentric circles************" <<endl;
	vector<ConcentricCircles> circs;
	//resX, resY, resR determine the resolution in the direction X,Y and R respectively
	int min_radius=15,max_radius=150,thres=400,resX=6,resY=6,resR=7;
	detectConcentric(edges, frame_gray.size(), min_radius, max_radius, thres, resX, resY, resR, circs);
	cout<<"Found "<< circs.size() << " concentric circles:" << endl;
	for(vector<ConcentricCircles>::iterator it = circs.begin();it!=circs.end();++it){
		cout << "("<<(*it).xc <<", "<<(*it).yc<<"): ";
		for(int i=0;i<(*it).num;i++){
			cout << (*it).rs[i] << " ";
		}
		cout << " Score: "<<(*it).total_acc<<endl; //Print total score for each concentric circle
	}
	cout << "**************************************************" << endl<<endl;


// 5. Create bounding boxes for circles that scored a high accumulator value. Submit for further analysis otherwise
  cout << "*******Creating bounding boxes using circles******" <<endl;
	int confidence_thresh = 1000;
  for(vector<ConcentricCircles>::iterator it = circs.begin();it!=circs.end();++it){
		//Point used as an offset from the center for top-left and bottom-right points of the rectangle
		Point offset((*it).rs[(*it).num-1],(*it).rs[(*it).num-1]);
		Point center((*it).xc,(*it).yc);
		//Create new bounding box using the largest radius found
		Rect box(center-offset,center+offset);
		//If the total score is greater than "confidence_thresh"
		//then this is a front-facing dartboard --> use newly created bounding box
		//A bounding box created using this method will skip the rest of the detection procedure
		if((*it).total_acc>confidence_thresh)
			finalOut.push_back(box);
		else //otherwise add the center of the circle to the list of possible dartboards
			candidate_centers.push_back(center);
	}
	cout << "Circles classified as dartboards: " << finalOut.size()<<endl;
	cout << "Circle centers submitted for further analysis: " << candidate_centers.size() << endl;
	cout << "**************************************************"<<endl<<endl;


// 6. Remove related Viola-Jones boxes
  cout << "*********Removing nearby Viola-Jones boxes********"<<endl;
	int dist_thresh = 80;
	int counter = 0;
	//For each bounding box created in stage 5, remove all bounding boxes
	//of Viola-Jones within a distance of "dist_thresh"(and those with an area of intersection)
	//because we do not need them anymore.
	for(vector<Rect>::iterator it=finalOut.begin();it!=finalOut.end();++it){
		Rect accepted_box = *it;
		Point accepted_boxC(accepted_box.tl().x+accepted_box.width/2,accepted_box.tl().y+accepted_box.height/2);
		vector<Rect>::iterator violaBox = output.begin();
		while(violaBox!=output.end()){
			Rect box = *violaBox;
			Point boxC(box.tl().x+box.width/2,box.tl().y+box.height/2);
			float dist = norm(accepted_boxC-boxC);
			double interArea = (box & accepted_box).area();
			if(interArea>0 || dist<dist_thresh){
				violaBox = output.erase(violaBox);
				counter++;
			}
			else{
				++violaBox;
			}
		}
	}
	cout << "Comparing circle boxes with Viola-Jones boxes..."<<endl;
	cout << "Viola-Jones boxes removed: " << counter <<endl;
	cout << "Viola-Jones boxes left: " << output.size() << endl;
	cout << "**************************************************" <<endl<<endl;


  #ifndef NO_ELLIPSE_DETECTION


// 7. Hough lines
	cout << "****Performing Hough transform to detect lines****" << endl;
	//Remove bounding boxes which do not include a lot of lines
  HoughLinesFilter(frame_gray, output);
	cout << "Bounding boxes left: " << output.size() << endl;
	cout << "**************************************************" << endl<<endl;

	//If no bounding boxes left, then do not perform
	//ellipse detection
	if(output.size() == 0){
		output = finalOut;
		return;
	}

// 8. Extract edges as a 1D array for Ellipses
	cout << "*********Preparing for ellipse detection**********" << endl;
	vector<EdgePointInfo> edgesEllipses;
	//Extract only the edges that are contained within bounding boxes
	extractEdges(frame_gray, output, edgesEllipses, 1, 70, 5);
	cout<<"Edge pixels found for ellipse detection: "<<edgesEllipses.size()<<endl;
	cout << "**************************************************" << endl<<endl;

// 7. Detect ellipses
  cout << "**Detecting ellipses. This might take a while...**" <<endl;
  Mat frame_gray_copy = frame_gray.clone();
	vector<MyEllipse> ellipses;
  int threshold = 35, minMajor = 50, maxMajor = 200;
  detectEllipse(edgesEllipses, frame_gray.size(), ellipses, threshold, minMajor, maxMajor);
	cout<<"Ellipses found: "<<ellipses.size()<<endl;
	for(unsigned int i=0;i<ellipses.size();i++){
		cout<<"("<<ellipses[i].xc<<", "<<ellipses[i].yc<<"): angle("<<ellipses[i].angle*180/pi<<") major("<<ellipses[i].major<<") minor("<<ellipses[i].minor<<") Score: "<<ellipses[i].accum<<endl;
		#ifdef DEBUG
		ellipse(frame_gray_copy, Point(ellipses[i].xc,ellipses[i].yc),Size(ellipses[i].major,ellipses[i].minor),ellipses[i].angle*180/pi,0,360,Scalar(0,255,0),2);
		#endif
	}
	#ifdef DEBUG
	namedWindow("Ellipse Detection",CV_WINDOW_AUTOSIZE);
	imshow("Ellipse Detection", frame_gray_copy);
	waitKey(0);
	#endif
	cout << "***************************************************" <<endl<<endl;

// 8. Accumulate all centers
  cout << "***********Accumulating all centers found**********" <<endl;
	//The center of ellipses are collected together with the centers
	//of concentric circles(those that failed the confidence test)
	for(vector<MyEllipse>::iterator ell=ellipses.begin();ell!=ellipses.end();++ell){
		Point ellC( (*ell).xc, (*ell).yc );
		candidate_centers.push_back(ellC);
	}
	cout << "Found " << candidate_centers.size() << " additional possible dartboard centers"<< endl;
	cout << "***************************************************" <<endl<<endl;


// 9. Merge or discard close centers
  cout << "*************Merging remaining centers*************" << endl;
  vector<Point> centers;
	int minDist = max(frame_gray.size().height,frame_gray.size().width)/5;
	//finalOut contains confirmed dartboards, therefore discard all candidate centers
	//that are nearby to confirmed centers
	for(vector<Rect>::iterator boxIt = finalOut.begin();boxIt!=finalOut.end();++boxIt){
		Point rectC((*boxIt).tl().x+(*boxIt).width/2, (*boxIt).tl().y+(*boxIt).height/2);
		vector<Point>::iterator it = candidate_centers.begin();
		while(it!=candidate_centers.end()){
			double dist = norm((*it)-rectC);
			if(dist<minDist){
				it = candidate_centers.erase(it);
			}
			else{
				++it;
			}
		}
	}
	//Check all remaining candidate centers and merge the
	//centers that are close to each other by using their average as a center
	while(candidate_centers.size()!=0)
	{
		Point cent = mergeCloseCenters(candidate_centers, frame_gray.size());
		centers.push_back(cent);
	}
	cout<<"Centers remaining after merging: "<<centers.size()<<endl;
	for(size_t i=0;i<centers.size();i++){
		cout<<i<<". "<<centers[i]<<endl;
	}
	cout << "***************************************************" <<endl<<endl;

  #endif

	#ifdef NO_ELLIPSE_DETECTION
	vector<Point> centers = candidate_centers;
	#endif

// 10. Classify bounding boxes based on nearby centers
  cout << "*Classifying bounding boxes based on found centers*" << endl;
  int distance_thr = 80;
	int dart_mask[output.size()] = {0};
	//vector<Point> minCenters
	//Decide to which center does each bounding box belong to
	for(unsigned int i = 0; i<output.size();++i){
		dart_mask[i] = -1;
		float minDist = 9999999;
		int minDistIndex = 0;
		for(unsigned int j=0;j<centers.size();j++){
			Point rectC( output[i].x+output[i].width/2, output[i].y+output[i].height/2 );
			Point circC( centers[j] );
			double dist = norm(rectC-circC);
			if(dist<minDist){ //find the closest center
				minDist = dist;
				minDistIndex = j;
			}
		}
		//assign that bounding box to that center if the distance is less than a threshold
		if(minDist<distance_thr) dart_mask[i] = minDistIndex;
	}
	int boxes_left = 0;
	cout<<"Class per bounding box:  ";
	for(size_t i=0;i<output.size();i++){
	  cout<<dart_mask[i]<<" ";
		if (dart_mask[i] >= 0) boxes_left++; //Count how many boxes where left after classification
	}
	cout<<endl;
	cout<<"Bounding boxes left after classification: "<<boxes_left<<endl;
	cout << "***************************************************" <<endl<<endl;

// 11. Merge bounding boxes per cluster
  cout << "************Merging close bounding boxes***********" <<endl;
	//If there are 2 or more bounding boxes assigned to a cluster,
	//merge them using a weighted average(based on distance from center)
  for(size_t j=0;j<centers.size();j++){
		double tlX=0;
		double tlY=0;
		double brX=0;
		double brY=0;
		double sumWeight = 0;
		bool matchFound = false;
		for(unsigned int i=0;i<output.size();i++){
			if(dart_mask[i]==(int)j){
				matchFound = true; //Center has bounding boxes assigned to it
				Point rectC( output[i].x+output[i].width/2, output[i].y+output[i].height/2 );
				Point circC( centers[j] );
				double dist = norm(rectC-circC);
				double weight = 1.0f/dist; //inverse proportional to distance
				sumWeight += weight; //Calculate total weight
				//Calculate top left and bottom right coordinates
				tlX += weight*output[i].tl().x;
				tlY += weight*output[i].tl().y;
				brX += weight*output[i].br().x;
				brY += weight*output[i].br().y;
			}
		}
		if(matchFound)
		{
			Point newTL(tlX/sumWeight,tlY/sumWeight);
			Point newBR(brX/sumWeight,brY/sumWeight);
	    Rect box(newTL, newBR); //Create new rectange
			finalOut.push_back(box);
	  }
	}
	output = finalOut;
	cout << "Bounding boxes left after merging: "<<output.size()<<endl;
	cout << "***************************************************" << endl<<endl;

}

Point mergeCloseCenters(vector<Point>& candidate_centers, Size imsize)
{
	Point c(*candidate_centers.begin());
	candidate_centers.erase(candidate_centers.begin());
	int dist_thresh = max(imsize.height,imsize.width)/5;
	vector<Point>::iterator it = candidate_centers.begin();
	Point avg(c);
	int counter = 1;
	while(it!=candidate_centers.end())
	{
		double dist = norm(c-*(it));
		if(dist<dist_thresh)
		{
			avg += (*it);
			counter++;
			it = candidate_centers.erase(it);
		}
		else{
			++it;
		}
	}
	avg = avg/counter;
	return avg;
}

void HoughLinesFilter(const Mat& frame_gray, vector<Rect>& output)
{
	Mat src_gray = frame_gray.clone();
	Mat edges;

  GaussianBlur( src_gray, src_gray, Size(5,5), 0, 0, BORDER_DEFAULT );

	int kernel = 3;
	int ratio = 3;
	int low_threshold=70;

	Canny(src_gray, edges, low_threshold, low_threshold*ratio, kernel, true);
	//dilate(edges,edges,Mat());
	#ifdef DEBUG
	namedWindow("Hough Edges", CV_WINDOW_AUTOSIZE);
	imshow("Hough Edges",edges);
	waitKey(0);
	#endif

	vector<Vec4i> lines; //vector holding lines to be detected
	//HoughLinesP(edges, lines, 3, 1*CV_PI/180, 70, 15, 10);
	//HoughLinesP(edges, lines, 1, 2*CV_PI/180, 50, 15, 20);
	HoughLinesP(edges, lines, 3, 3*CV_PI/180, 65, 25, 8);//15,15,5 works

	vector<Point> midPoints; // vector holding line midpoints
	Point mid; //line midpoint
	for(size_t i=0 ; i<lines.size(); i++ ){
		line(src_gray, Point(lines[i][0], lines[i][1]), Point(lines[i][2],lines[i][3]),Scalar(0,255,0),1,8); //draw line
		mid = Point((lines[i][0]+lines[i][2])/2 ,(lines[i][1]+lines[i][3])/2);
		circle( src_gray, mid, 3,  Scalar(0), 2, 8, 0 );
		midPoints.push_back(mid);
		//cout<<midPoints[i]<<endl;
	}
	#ifdef DEBUG
	namedWindow("HoughLines",CV_WINDOW_AUTOSIZE);
	imshow("HoughLines",src_gray);
	waitKey(0);
	#endif

	// double midThreshold = 0.002;
	double midThreshold = 0.0025;//0.0015
	//The score required for each bounding box is based
	//on the area of the box
	vector<Rect>::iterator it = output.begin();
	cout<<"Lines found: "<<midPoints.size()<<". Calculating scores: "<<endl;
	while(it!=output.end()){
		int midScore = 0;
		for(size_t j=0; j<midPoints.size(); j++){
			if((*it).contains(midPoints[j])) midScore++;
		}
		double required = max(5.0, midThreshold * (*it).area()); //minimum required score 5
		cout<<(*it)<<"  \tScore: "<<midScore<<". Required: "<<(int)required<<endl;
		if(midScore<(int)required) it=output.erase(it);
		else ++it;
	}
}

double rectIntersection(Rect A, Rect B){
	Rect C =  A & B;
	return (float)C.area() / (float)(A.area()+B.area()-C.area());
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
	double thresh = 0.3;
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

  int FP = detected.size(); //remaining rectangles are false positives
	int FN = ground.size(); //remaining ground truths are false negatives

	return (double)2*TP / (double)(2*TP+FN+FP);
}

bool valid(int num, int min, int max){
  if(num<min)
    return false;
  if(num>=max)
    return false;
  return true;
}

/*
 * Ellipse detection inspired from ”A New Efficient Ellipse Detection Method” (Yonghong Xie
 * Qiang , Qiang Ji / 2002).
 * Parts of the code are ported from the matlab function http://uk.mathworks.com/matlabcentral/fileexchange/33970-ellipse-detection-using-1d-hough-transform
 */
void detectEllipse(vector<EdgePointInfo>& edgeList, Size imsize, vector<MyEllipse>& output, int threshold, int minMajor, int maxMajor)
{
	double eps = 0.0001; //machine floating point error
	//int rotationSpan = 90;
	float minAspectRatio = 0.35;  //major/minor must be greater than this
	size_t edgeNum = edgeList.size();
	int min_distance = max(imsize.height,imsize.width)/5;

	cout<<"Possible major axes: "<<edgeNum*edgeNum<<endl;

  vector<int> I; //I.reserve(edgeNum*edgeNum);
	vector<int> J; //J.reserve(edgeNum*edgeNum);
	vector<int> distsSq;// distsSq.reserve(edgeNum*edgeNum);

	for(size_t i=0;i<edgeNum;i++){
		for(size_t j=i+1;j<edgeNum;j++){
			//cout<<i<<" "<<j<<endl;
			int x1 = edgeList[i].x;
			int x2 = edgeList[j].x;
			int y1 = edgeList[i].y;
			int y2 = edgeList[j].y;

			int length = (x2-x1)*(x2-x1) + (y2-y1)*(y2-y1);

			//Collect all pairs of edges that satisfy the length constraints.
			//Assume that those are possible major axes of an ellipse
			if(length<(maxMajor*maxMajor) && length>(minMajor*minMajor))
			{

				I.push_back(i);
				J.push_back(j);
				distsSq.push_back(length);
			}
		}
	}

  size_t npairs = I.size();

	cout<<"Possible major axes after length constraint: "<<npairs<<endl;

  //ANGULAR CONSTRAINT HERE IF NEEDED

  int randomise = 4;
	int* perm = new int[npairs]; //allocate memory

	//Fill the array in parallel.
	//This holds the indices for the pairs
	#pragma omp parallel for
	for(size_t i = 0; i < npairs; i++) {
		perm[i] = i;
	}

  size_t subset_max = npairs;
	if(randomise>0){
		//Shuffle the indices and pick the first "subset_max" indices to use
		random_shuffle( &perm[0], &perm[npairs] ); //MergeShuffle can be used here to speed this up
		subset_max = min(npairs, randomise * edgeNum); //Use randomise*edgeNum pairs of points
	}

	cout<<"Possible major axes after randomisation: "<<subset_max<<endl;

	for(size_t i = 0;i<subset_max;i++)
	{
		int x1 = edgeList[I[perm[i]]].x, y1 = edgeList[I[perm[i]]].y;
		int x2 = edgeList[J[perm[i]]].x, y2 = edgeList[J[perm[i]]].y;
		int x0 = (x1+x2)/2, y0=(y1+y2)/2; //Calculate center
		float a_sq = distsSq[perm[i]]/4.0f; //Major *radius* squared

		Mat accum = Mat::zeros(1,&maxMajor,CV_16U);

		#pragma omp parallel
		{
		float thirdPointDist_sq; //distance from a third point
		float f_sq; //distance of third point and (x2,y2)
		float costau; //cos of tau(tau being the angle from the major axis to the third point)
		float sintau_sq; //sin of tau squared
		float b; //minor radius
		int bin; //minor radius rounded to match a bin in the accumulator array
		vector<int> local_accum;
		local_accum.resize(maxMajor,0);
		//loop over all edge pixels(in parallel) and calculate parameters of candidate ellipse
		#pragma omp for private(thirdPointDist_sq,f_sq,costau,sintau_sq,b,bin)
		for(size_t k = 0; k < edgeNum; k++) {
			EdgePointInfo p = edgeList[k];
			thirdPointDist_sq = (p.x-x0)*(p.x-x0) + (p.y-y0)*(p.y-y0);
			if(thirdPointDist_sq <= a_sq){
				f_sq = (p.x-x2)*(p.x-x2) + (p.y-y2)*(p.y-y2);
				//Cosine rule
				costau = ( a_sq + thirdPointDist_sq - f_sq ) / ( 2*sqrt(a_sq*thirdPointDist_sq) );
				costau = min(1.0f, max(-1.0f, costau)); //clip between -1 and 1
				sintau_sq = 1-costau*costau;
				b = sqrt( (a_sq * thirdPointDist_sq * sintau_sq) / (a_sq - thirdPointDist_sq * costau * costau + eps) );
				bin = ceil(b+eps);
				local_accum[bin]++; //vote in a local accumulator
			}
		}
		//collect all votes in a global accumulator(critical region)
		#pragma omp critical
		{
			short int* acc_ptr = accum.ptr<short int>();
			for(int n=0;n<maxMajor;n++){
				acc_ptr[n] += local_accum[n];
			}
		}
	  }
		GaussianBlur( accum, accum, Size(1,5), 1, 0, BORDER_DEFAULT ); //apply some smoothing
		short int* acc_ptr = accum.ptr<short int>();
		short int max = 0;
		short int maxIndex = 0;
		//Find the minor radius with the most votes
		for(int l=maxMajor-1;l>=ceil(sqrt(a_sq)*minAspectRatio);l--){
			//cout<<max<<endl;
			if(acc_ptr[l]>max){
				max = acc_ptr[l];
				maxIndex = l;
			}
			acc_ptr[l] = 0;
		}

		if(max>threshold){ //Create ellipse
			MyEllipse candidate(x0, y0, atan2(y1-y2,x1-x2), sqrt(a_sq), maxIndex, max);
			vector<MyEllipse>::iterator it = output.begin();
			bool confirmed = true;
			while(it!=output.end())
			{
				if(min_distance*min_distance  > ((*it).xc-candidate.xc)*((*it).xc-candidate.xc)+((*it).yc-candidate.yc)*((*it).yc-candidate.yc) )
				{
					if((*it).accum > candidate.accum){ //if there is an ellipse within a distance with higher score then keep that
						confirmed = false;
						++it;
					}
					else{ //otherwise keep this one and delete the other ellipse
						it = output.erase(it);
					}
				}
				else
				{
					++it;
				}
			}
			if(confirmed){
				output.push_back(candidate);
			}
		}
	}

	delete(perm); //free memory

}

void detectConcentric(vector<EdgePointInfo>& edgeList, Size imsize, int min_radius, int max_radius,
											int threshold, int resX, int resY, int resR, vector<ConcentricCircles>& output)
{
	int min_distance = max(imsize.height,imsize.width)/5;
	int vote_around_radius = 0; //measured in bins

  int ndims = 3;
	double low_threshold = threshold * 0.4;
	int sizes[3] = { imsize.height/resY, imsize.width/resX, (max_radius-min_radius)/resR };

	Mat accum;
	accum = Mat::zeros(ndims, sizes, CV_16U);
	//for(int i=0;i<accum.dims;i++)
	//cout<<accum.size[i]<<endl;

	for(vector<EdgePointInfo>::iterator it = edgeList.begin(); it!=edgeList.end(); ++it){
		EdgePointInfo pt = *it;
		//cout << pt.x <<" "<< pt.y<< endl;
		double cospt = cos(pt.grad);
		double sinpt = sin(pt.grad);
		//cout<<pt.x<<" "<<pt.y<<" "<<pt.grad<<" "<<cospt<<" "<<sinpt<<endl;

		for(int i=min_radius;i<max_radius;i++){
			int x0a = (int)(pt.x + i*cospt);
			int y0a = (int)(pt.y + i*sinpt);
			int x0b = (int)(pt.x - i*cospt);
			int y0b = (int)(pt.y - i*sinpt);
			int y_pos_a = y0a/resY;
			int x_pos_a = x0a/resX;
			int x_pos_b = x0b/resX;
			int y_pos_b = y0b/resY;

			for(int q=-vote_around_radius;q<=vote_around_radius;q++)
			{
				for(int w=-vote_around_radius;w<=vote_around_radius;w++)
				{
					if(valid(x_pos_a+q,0,sizes[1]) && valid(y_pos_a+w,0,sizes[0]) )
						accum.at<short int>(y_pos_a+w,x_pos_a+q,(i-min_radius)/resR) += 1;
					if(valid(x_pos_b+q,0,sizes[1]) && valid(y_pos_b+w,0,sizes[0]) )
						accum.at<short int>(y_pos_b+w,x_pos_b+q,(i-min_radius)/resR) += 1;
				}
			}
		}
	}

	for(int y = 1;y<sizes[0]-1;y++)
	{
		for(int x=1;x<sizes[1]-1;x++)
		{
			ConcentricCircles c;
			c.num = 0;
			c.total_acc = 0;
			c.xc = cvRound((x+0.5f)*resX);
			c.yc = cvRound((y+0.5f)*resY);
			bool found_at_least_one = false;
			for(int r=0;r<sizes[2];r++)
			{
				//The center of a circle must be greater than the threshold and also must be the maximum from its
				//neighbours. i.e. no center to the left,top,bottom or right can be better
				//We also require for at least one circle in this (x,y) to be greater than the threshold
				//but the rest of the circles are found using a lower threshold to promote the detection of concentric circles
				if(accum.at<short int>(y,x,r) > threshold &&
					 accum.at<short int>(y,x,r) > accum.at<short int>(y,x-1,r) && accum.at<short int>(y,x,r) > accum.at<short int>(y,x+1,r) &&
					 accum.at<short int>(y,x,r) > accum.at<short int>(y-1,x,r) && accum.at<short int>(y,x,r) > accum.at<short int>(y+1,x,r) )
				{
					found_at_least_one = true;
					break;
				}
			}
			if(found_at_least_one)
			{
				for(int r=0;r<sizes[2];r++)
				{
					if(accum.at<short int>(y,x,r) > low_threshold && //use lower threshold now
				     accum.at<short int>(y,x,r) > accum.at<short int>(y,x-1,r) && accum.at<short int>(y,x,r) > accum.at<short int>(y,x+1,r) &&
					 	 accum.at<short int>(y,x,r) > accum.at<short int>(y-1,x,r) && accum.at<short int>(y,x,r) > accum.at<short int>(y+1,x,r) ) //might need to add for radius too
					{
						//cout<<accum.at<short int>(y,x,r)<<endl;
						if(c.num < MAX_CONCENTRIC){
							c.rs[c.num++] = cvRound(min_radius+(r+0.5f)*resR);
							c.total_acc += accum.at<short int>(y,x,r);
						}
					}
				}
		  }
			if(c.num>=2) //at least 2 circles to be considered concentric
			{
				vector<ConcentricCircles>::iterator cir = output.begin();
				bool confirmed = true;
				while(cir!=output.end())
				{
					if(min_distance*min_distance  > ((*cir).xc-c.xc)*((*cir).xc-c.xc)+((*cir).yc-c.yc)*((*cir).yc-c.yc) )
					{
						if((*cir).total_acc > c.total_acc){
							confirmed = false;
							++cir;
						}
						else{
							cir = output.erase(cir);
						}
					}
					else
					{
						++cir;
					}
				}
				if(confirmed){
					output.push_back(c);
				}
			}
		}
	}
	#ifdef DEBUG
	show3Dhough(accum);
	#endif
}

void show3Dhough(Mat& input){

	if(input.dims!=3){
		cout<<"Only 3 dimensional matrices are accepted!"<<endl;
		return;
	}

	int dimX = input.size[1];
	int dimY = input.size[0];
	int dimZ = input.size[2];
	int sizes[2] = {dimY,dimX};

	Mat output;
	output.create(2,sizes,CV_8U);

	for(int y=0; y<dimY; y++){
		for(int x=0; x<dimX; x++){
			int total = 0;
			for(int z=0; z<dimZ; z++){
				 total += input.at<short int>(y,x,z);
			}
			output.at<uchar>(y,x) = saturate_cast<uchar>(log(total)*25);
		}
	}
	namedWindow("Hough Space",CV_WINDOW_AUTOSIZE);
	imshow("Hough Space", output);
	waitKey(0);
}

void extractEdges(Mat& gray_input, vector<Rect>& v, vector<EdgePointInfo>& edgeList, int method, int edge_thresh, int kernel_size)
{
  float magScale = 255.f/1442.f;
	int scale = 1;
	int delta = 0;
	Mat src = gray_input.clone();
	Mat dy, dx;
	Mat thres, grad;

	thres.create(src.size(),src.type());
	grad.create(src.size(),src.type());

	GaussianBlur( src, src, Size(kernel_size,kernel_size), 0, 0, BORDER_DEFAULT );

	Sobel( src, dx, CV_16S, 1, 0, 3, scale, delta, BORDER_DEFAULT );
	Sobel( src, dy, CV_16S, 0, 1, 3, scale, delta, BORDER_DEFAULT );

  int nRows = src.rows;
	int nCols = src.cols;

  if(method == 0)
	{
		for(int i=0;i<nRows;++i)
		{
			short int* dx_ptr = dx.ptr<short int>(i);
			short int* dy_ptr = dy.ptr<short int>(i);
			uchar* thres_ptr  = thres.ptr<uchar>(i);
			uchar* grad_ptr    = grad.ptr<uchar>(i);
	    for(int j = 0;j<nCols;j++)
			{
				double psi = atan2((double)dy_ptr[j]/8.0,(double)dx_ptr[j]/8.0);
				grad_ptr[j] = (uchar)255*(psi-(-pi))/(2*pi);
				double mag = magScale*sqrt(dx_ptr[j]*dx_ptr[j]+dy_ptr[j]*dy_ptr[j]);
				Point epiPoint(j,i);
				EdgePointInfo epi;
				epi.x = j;
				epi.y = i;
				epi.grad = psi;
				if(mag > edge_thresh){
					vector<Rect>::iterator it;
					for(it = v.begin(); it!= v.end(); ++it){
						if((*it).contains(epiPoint)){
							thres_ptr[j] = 255;
							edgeList.push_back(epi);
							break;
						}
					}
					if(it==v.end())
						thres_ptr[j] = 0;
					if(v.size()==0) // If mask is empty, include the edge anyway
					{
						thres_ptr[j] = 255;
						edgeList.push_back(epi);
					}
				}
				else{
					thres_ptr[j] = 0;
				}
			}
		}
  }
  else{
		int ratio = 3; //ratio between lower and upper thresholds
		int kernel_size = 3;
		Canny( src,  thres, edge_thresh, edge_thresh*ratio, kernel_size, true );
	  for(int i=0;i<nRows;++i)
		{
			short int* dx_ptr = dx.ptr<short int>(i);
			short int* dy_ptr = dy.ptr<short int>(i);
			uchar* thres_ptr  = thres.ptr<uchar>(i);
			uchar* grad_ptr    = grad.ptr<uchar>(i);
	    for(int j = 0;j<nCols;j++)
			{
				double psi = atan2((double)dy_ptr[j]/8.0,(double)dx_ptr[j]/8.0);
				grad_ptr[j] = (uchar)255*(psi-(-pi))/(2*pi);
				if(thres_ptr[j]==255){
					EdgePointInfo epi;
					epi.x = j;
					epi.y = i;
					epi.grad = psi;
					Point epiPoint(epi.x,epi.y);
					vector<Rect>::iterator it;
					for(it = v.begin(); it!= v.end(); ++it){
						if((*it).contains(epiPoint)){
							edgeList.push_back(epi);
							break;
						}
					}
					if(it==v.end())
						thres_ptr[j] = 0;
					if(v.size()==0) // If mask is empty, include the edge anyway
					{
						thres_ptr[j] = 255;
						edgeList.push_back(epi);
					}
				}
			}
		}
  }
	#ifdef DEBUG
	namedWindow("Gradient",CV_WINDOW_AUTOSIZE);
	imshow("Gradient", grad);
	namedWindow("Thresholded magnitude",CV_WINDOW_AUTOSIZE);
	imshow("Thresholded magnitude", thres);
	waitKey(0);
	#endif

}
