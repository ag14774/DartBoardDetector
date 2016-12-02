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
//#define ONLY_CONCENTRIC_CIRCLES

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

typedef struct{
	int x;
	int y;
	double grad;
}EdgePointInfo;

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
	int xc;
	int yc;
	float angle;
	int major;
	int minor;
	float accum;

}MyEllipse;

/** Function Headers */
void detect( Mat& frame, vector<Rect>& output );
void drawRects(Mat& frame, Mat& output, vector<Rect> v);
void HoughLinesFilter(const Mat& frame_gray, vector<Rect>& output);
                                                                           //inverse of resolution in X, Y and R direction
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
	srand(time(0));

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
  cout<<"Number of dartboards found: "<<output.size()<<endl<<endl;
	// 4. Draw result
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

/** @function detectAndDisplay */
void detect( Mat& frame, vector<Rect>& output )
{
	Mat frame_gray;
	Mat frame_gray_norm;
	Mat frame_blur,frame_dst;
	Mat frame_gray2;
	vector<Rect> finalOut;
	vector<Point> candidate_centers;


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
	vector<Rect> dummy;
  vector<EdgePointInfo> edges;
	extractEdges(frame_gray, dummy, edges, 1, 50, 5);
	cout<<"Edge pixels found for circle detection: "<<edges.size()<<endl;
	cout << "**************************************************" << endl<<endl;


// 4. Detect concentric circles
  cout << "**********Detecting concentric circles************" <<endl;
	vector<ConcentricCircles> circs;
	int min_radius=15,max_radius=150,thres=400,resX=6,resY=6,resR=7;
	detectConcentric(edges, frame_gray.size(), min_radius, max_radius, thres, resX, resY, resR, circs);
	cout<<"Found "<< circs.size() << " concentric circles:" << endl;
	for(vector<ConcentricCircles>::iterator it = circs.begin();it!=circs.end();++it){
		cout << "("<<(*it).xc <<", "<<(*it).yc<<"): ";
		for(int i=0;i<(*it).num;i++){
			cout << (*it).rs[i] << " ";
		}
		cout << " Score: "<<(*it).total_acc<<endl;
	}
	cout << "**************************************************" << endl<<endl;


// 5. Create bounding boxes from circles
  cout << "*******Creating bounding boxes using circles******" <<endl;
	int confidence_thresh = 1000;
  for(vector<ConcentricCircles>::iterator it = circs.begin();it!=circs.end();++it){
		Point offset((*it).rs[(*it).num-1],(*it).rs[(*it).num-1]);
		Point center((*it).xc,(*it).yc);
		Rect box(center-offset,center+offset);
		if((*it).total_acc>confidence_thresh)
			finalOut.push_back(box);
		else
			candidate_centers.push_back(center);
	}
	cout << "Circles classified as dartboards: " << finalOut.size()<<endl;
	cout << "Circle centers submitted for further analysis: " << candidate_centers.size() << endl;
	cout << "**************************************************"<<endl<<endl;

	#ifdef ONLY_CONCENTRIC_CIRCLES
	output=finalOut;
	return;
	#endif


// 6. Remove related Viola-Jones boxes
  cout << "*********Removing nearby Viola-Jones boxes********"<<endl;
	int dist_thresh = 80;
	int counter = 0;
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


// 7. Hough lines
	cout << "****Performing Hough transform to detect lines****" << endl;
  //vector<Rect> output2 = output;
  HoughLinesFilter(frame_gray, output);
	cout << "Bounding boxes left: " << output.size() << endl;
	//if(output.size()==0){
	//	cout<<"Hough Lines failed to detect any dartboards. Undoing..."<<endl;
	//	output=output2;
	//	cout << "Bounding boxes left after hough lines: "<<output.size() << endl;
	//}
	cout << "**************************************************" << endl<<endl;

	if(output.size() == 0){
		output = finalOut;
		return;
	}

// 8. Extract edges as a 1D array for Ellipses
	cout << "*********Preparing for ellipse detection**********" << endl;
	vector<EdgePointInfo> edgesEllipses;
	extractEdges(frame_gray, output, edgesEllipses, 1, 70, 5);
	cout<<"Edge pixels found for ellipse detection: "<<edgesEllipses.size()<<endl;
	cout << "**************************************************" << endl<<endl;

// 7. Detect ellipses
  cout << "**Detecting ellipses. This might take a while...**" <<endl;
  Mat frame_gray_copy = frame_gray.clone();
	vector<MyEllipse> ellipses;
	//int threshold = 50, minMajor = 15, maxMajor = 200;
  //detectEllipse(edgesAll, frame_gray.size(), ellipses, threshold, minMajor, maxMajor);
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


// 10. Cluster bounding boxes based on nearby centers
  cout << "*Classifying bounding boxes based on found centers*" << endl;
  int distance_thr = 80;
	int dart_mask[output.size()] = {0};
	//vector<Point> minCenters
	for(unsigned int i = 0; i<output.size();++i){
		dart_mask[i] = -1;
		float minDist = 9999999;
		int minDistIndex = 0;
		for(unsigned int j=0;j<centers.size();j++){
			Point rectC( output[i].x+output[i].width/2, output[i].y+output[i].height/2 );
			Point circC( centers[j] );
			double dist = norm(rectC-circC);
			if(dist<minDist){
				minDist = dist;
				minDistIndex = j;
			}
		}
		if(minDist<distance_thr) dart_mask[i] = minDistIndex;
	}
	int boxes_left = 0;
	cout<<"Class per bounding box:  ";
	for(size_t i=0;i<output.size();i++){
	  cout<<dart_mask[i]<<" ";
		if (dart_mask[i] >= 0) boxes_left++;
	}
	cout<<endl;
	cout<<"Bounding boxes left after classification: "<<boxes_left<<endl;
	cout << "***************************************************" <<endl<<endl;

// 11. Merge bounding boxes per cluster
  cout << "************Merging close bounding boxes***********" <<endl;

  for(size_t j=0;j<centers.size();j++){
		double tlX=0;
		double tlY=0;
		double brX=0;
		double brY=0;
		double sumWeight = 0;
		bool matchFound = false;
		for(unsigned int i=0;i<output.size();i++){
			if(dart_mask[i]==(int)j){
				matchFound = true;
				Point rectC( output[i].x+output[i].width/2, output[i].y+output[i].height/2 );
				Point circC( centers[j] );
				double dist = norm(rectC-circC);
				double weight = 1.0f/dist; //inverse proportional to distance
				sumWeight += weight;
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
	    Rect box(newTL, newBR);
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

  //blur( src_gray, src_gray, Size(3,3) );

	//threshold(src_gray,src_gray,120,255,0);

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
	vector<Rect>::iterator it = output.begin();
	cout<<"Lines found: "<<midPoints.size()<<". Calculating scores: "<<endl;
	while(it!=output.end()){
		int midScore = 0;
		for(size_t j=0; j<midPoints.size(); j++){
			if((*it).contains(midPoints[j])) midScore++;
		}
		double required = max(5.0, midThreshold * (*it).area());
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

  int FP = detected.size();
	int FN = ground.size();

	return (double)2*TP / (double)(2*TP+FN+FP);
}

bool valid(int num, int min, int max){
  if(num<min)
    return false;
  if(num>=max)
    return false;
  return true;
}

void detectEllipse(vector<EdgePointInfo>& edgeList, Size imsize, vector<MyEllipse>& output, int threshold, int minMajor, int maxMajor)
{
	double eps = 0.0001;
	//int rotationSpan = 90;
	float minAspectRatio = 0.35;
	size_t edgeNum = edgeList.size();
	int min_distance = max(imsize.height,imsize.width)/5;

	cout<<"Possible major axes: "<<edgeNum*edgeNum<<endl;

  vector<int> I; //I.reserve(edgeNum*edgeNum);
	vector<int> J; //J.reserve(edgeNum*edgeNum);
	vector<int> distsSq;// distsSq.reserve(edgeNum*edgeNum);

	for(size_t i=0;i<edgeNum;i++){
		for(size_t j=i+1;j<edgeNum;j++){ //Inner loop can be vectorised
			//cout<<i<<" "<<j<<endl;
			int x1 = edgeList[i].x;
			int x2 = edgeList[j].x;
			int y1 = edgeList[i].y;
			int y2 = edgeList[j].y;

			int length = (x2-x1)*(x2-x1) + (y2-y1)*(y2-y1);

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

	#pragma omp parallel for
	for(size_t i = 0; i < npairs; i++) {
		perm[i] = i;
	}

  //CONSIDER THROWING A COIN FOR EACH PAIR
  size_t subset_max = npairs;
	if(randomise>0){
		random_shuffle( &perm[0], &perm[npairs] ); //Use MergeShuffle
		subset_max = min(npairs, randomise * edgeNum);
	}

	cout<<"Possible major axes after randomisation: "<<subset_max<<endl;

	for(size_t i = 0;i<subset_max;i++)
	{
		int x1 = edgeList[I[perm[i]]].x, y1 = edgeList[I[perm[i]]].y;
		int x2 = edgeList[J[perm[i]]].x, y2 = edgeList[J[perm[i]]].y;
		int x0 = (x1+x2)/2, y0=(y1+y2)/2;
		float a_sq = distsSq[perm[i]]/4.0f;

		Mat accum = Mat::zeros(1,&maxMajor,CV_16U);

		#pragma omp parallel
		{
		float thirdPointDist_sq;
		float f_sq;
		float costau;
		float sintau_sq;
		float b;
		int bin;
		vector<int> local_accum;
		local_accum.resize(maxMajor,0);
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
				local_accum[bin]++;
			}
		}
		#pragma omp critical
		{
			short int* acc_ptr = accum.ptr<short int>();
			for(int n=0;n<maxMajor;n++){
				acc_ptr[n] += local_accum[n];
			}
		}
	  }
		GaussianBlur( accum, accum, Size(1,5), 1, 0, BORDER_DEFAULT );
		short int* acc_ptr = accum.ptr<short int>();
		short int max = 0;
		short int maxIndex = 0;
		for(int l=maxMajor-1;l>=ceil(sqrt(a_sq)*minAspectRatio);l--){
			//cout<<max<<endl;
			if(acc_ptr[l]>max){
				max = acc_ptr[l];
				maxIndex = l;
			}
			acc_ptr[l] = 0;
		}
		//double predicted_circumference = 4*((sqrt(a_sq))+maxIndex)*pow(pi/4,4*sqrt(a_sq)*maxIndex/((sqrt(a_sq)+maxIndex)*(sqrt(a_sq)+maxIndex)));
		if(max>threshold){
			MyEllipse candidate(x0, y0, atan2(y1-y2,x1-x2), sqrt(a_sq), maxIndex, max);
			vector<MyEllipse>::iterator it = output.begin();
			bool confirmed = true;
			while(it!=output.end())
			{
				if(min_distance*min_distance  > ((*it).xc-candidate.xc)*((*it).xc-candidate.xc)+((*it).yc-candidate.yc)*((*it).yc-candidate.yc) )
				{
					if((*it).accum > candidate.accum){
						confirmed = false;
						++it;
					}
					else{
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
				if(accum.at<short int>(y,x,r) > threshold &&
					 accum.at<short int>(y,x,r) > accum.at<short int>(y,x-1,r) && accum.at<short int>(y,x,r) > accum.at<short int>(y,x+1,r) &&
					 accum.at<short int>(y,x,r) > accum.at<short int>(y-1,x,r) && accum.at<short int>(y,x,r) > accum.at<short int>(y+1,x,r) ) //might need to add for radius too
				{
					found_at_least_one = true;
					break;
				}
			}
			if(found_at_least_one)
			{
				for(int r=0;r<sizes[2];r++)
				{
					if(accum.at<short int>(y,x,r) > low_threshold &&
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
			if(c.num>=2) //at least 2 concentric circles
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
	// double mean = 0;
	// double std_deviation = 0;
	// for(size_t i=0;i<output.size();i++)
	// 	mean+= output[i].total_acc;
	// mean = mean / output.size();
	// for(size_t i=0;i<output.size();i++)
	// 	std_deviation += (output[i].total_acc-mean)*(output[i].total_acc-mean);
  // std_deviation = std_deviation / (output.size()-1);
	// double limit = mean - 1*std_deviation;
	// vector<ConcentricCircles>::iterator it = output.begin();
	// while(it!=output.end())
	// {
	// 	if((*it).total_acc<limit){
	// 		it = output.erase(it);
	// 	}
	// 	else{
	// 		++it;
	// 	}
	// }
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
		int ratio = 3;
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
