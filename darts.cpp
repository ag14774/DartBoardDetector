/////////////////////////////////////////////////////////////////////////////
//
// COMS30121 - darts.cpp
//
/////////////////////////////////////////////////////////////////////////////

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
#include <omp.h>
//#include "face_gt.h"
#include "darts_gt.h"

#define MAX_CONCENTRIC 20
#define DEBUG
#define GROUND_TRUTH
//#define ONLY_VIOLA_JONES

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
	MyEllipse(int nxc, int nyc, float nangle, int nmajor, int nminor, int naccum)
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
	int accum;

}MyEllipse;

/** Function Headers */
void detect( Mat& frame, vector<Rect>& output );
void drawRects(Mat& frame, Mat& output, vector<Rect> v);
void HoughLinesFilter(const Mat& frame_gray, vector<Rect>& output);
                                                                           //inverse of resolution in X, Y and R direction
void detectConcentric(vector<EdgePointInfo> edgeList, Size imsize, int min_radius, int max_radius,
											int threshold, int resX, int resY, int resR, vector<ConcentricCircles>& output);
void detectEllipse(vector<EdgePointInfo> edgeList, vector<MyEllipse>& output, int threshold, int minMajor, int maxMajor);
void extractEdges(Mat& gray_input, vector<Rect> v, vector<EdgePointInfo>& edgeList, int method, int edge_thresh);
double rectIntersection(Rect A, Rect B);
double fscore(vector<Rect> ground, vector<Rect> detected);

//******TO-DO*********
//1)Redo fscore and rectIntersection: Use intersection operator
//2)Review detector flowchart
//3)Implement ellipse detector
//4)Change step 7 of "detect" to choose the closest center instead of the first
//5)Function to convert 3D hough space to 2D

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


	// 1. Prepare Image by turning it into Grayscale
	cvtColor( frame, frame_gray, CV_BGR2GRAY );
	//Normalise lighting
  equalizeHist( frame_gray, frame_gray_norm );

	// 2. Perform Viola-Jones Object Detection
	cascade.detectMultiScale( frame_gray_norm, output, 1.1, 1, 0|CV_HAAR_SCALE_IMAGE, Size(50, 50), Size(500,500) );
	cout << "Dartboards found using Viola-Jones: "<<output.size()<<endl<<endl;

  #ifdef ONLY_VIOLA_JONES
	return;
	#endif

	// 3. Hough lines
  vector<Rect> output2 = output;
  HoughLinesFilter(frame_gray, output);
	cout << "Bounding boxes left: "<<output.size() << endl << endl;
	if(output.size()==0){
		cout<<"Hough Lines failed to detect any dartboards. Undoing..."<<endl;
		output=output2;
		cout << "Bounding boxes left: "<<output.size() << endl << endl;
	}

  vector<Rect> dummy;

// 4. Extract edges as a 1D array
  vector<EdgePointInfo> edges;
	//extractEdges(frame_gray, output, edges, 1, 30);
	extractEdges(frame_gray, dummy, edges, 1, 30);
	cout<<"Edge pixels found: "<<edges.size()<<endl<<endl;

  vector<MyEllipse> ellipses;
	int threshold = 120, minMajor = 10, maxMajor = 200;
  detectEllipse(edges, ellipses, threshold, minMajor, maxMajor);
	cout<<"Ellipses found: "<<ellipses.size()<<endl;
	for(int i=0;i<ellipses.size();i++){
		cout<<ellipses[i].xc<<" "<<ellipses[i].yc<<" "<<ellipses[i].angle*180/pi<<" "<<ellipses[i].accum<<endl;
		ellipse(frame, Point(ellipses[i].xc,ellipses[i].yc),Size(ellipses[i].major,ellipses[i].minor),ellipses[i].angle*180/pi,0,360,Scalar(0,255,0),2);
	}
return;
// 5. Detect concentric circles
	vector<ConcentricCircles> circs;
	int min_radius=15,max_radius=150,thres=300,resX=6,resY=6,resR=7;
	detectConcentric(edges, frame_gray.size(), min_radius, max_radius, thres, resX, resY, resR, circs);
	cout<<endl<<"Found "<< circs.size() << " concentric circles:" << endl;
	for(vector<ConcentricCircles>::iterator it = circs.begin();it!=circs.end();++it){
		cout << "("<<(*it).xc <<", "<<(*it).yc<<"): ";
		for(int i=0;i<(*it).num;i++){
			cout << (*it).rs[i] << " ";
		}
		cout << endl;
	}
	cout << endl;


// 6. Accumulate all centers
  vector<Point> centers;
	for(vector<ConcentricCircles>::iterator cir=circs.begin();cir!=circs.end();++cir){
		Point cirC( (*cir).xc, (*cir).yc );
		centers.push_back(cirC);
	}

// 7. Cluster bounding boxes based on nearby centers
  int distance_thr = 80;
	int dart_mask[output.size()] = {0};
	for(unsigned int i = 0; i<output.size();++i){
		dart_mask[i] = -1;
		for(unsigned int j=0;j<centers.size();j++){
			Point rectC( output[i].x+output[i].width/2, output[i].y+output[i].height/2 );
			Point circC( centers[j] );
			double dist = norm(rectC-circC);
			//int minDim = min(output[i].width, output[i].height);
			if(dist<distance_thr){
				dart_mask[i] = j;
				break;
			}
		}
	}
	cout<<"Clustering:  ";
	for(unsigned int i=0;i<output.size();i++)
	  cout<<dart_mask[i]<<" ";
	cout<<endl<<endl;

// 8. Merge bounding boxes per cluster
  vector<Rect> finalOut;
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
				double weight = 1.0/dist; //inverse proportional to distance
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

  /*for(int i=0;i<output.size();i++){
		if(dart_mask[i]>=0){
			Rect r(output[i]);
			finalOut.push_back(r);
		}
	}*/

	output = finalOut;

}

void HoughLinesFilter(const Mat& frame_gray, vector<Rect>& output)
{
	Mat src_gray = frame_gray.clone();
	Mat edges;

// 	/// Detector parameters
//  int blockSize = 2;
//  int apertureSize = 3;
//  double k = 0.04;
// Mat dst,dst_norm,dst_norm_scaled;
//  /// Detecting corners
//  cornerHarris( src_gray, dst, blockSize, apertureSize, k, BORDER_DEFAULT );
//  /// Normalizing
// normalize( dst, dst_norm, 0, 255, NORM_MINMAX, CV_32FC1, Mat() );
// convertScaleAbs( dst_norm, dst_norm_scaled );
//  /// Drawing a circle around corners
// for( int j = 0; j < dst_norm.rows ; j++ )
// 	 { for( int i = 0; i < dst_norm.cols; i++ )
// 				{
// 					if( (int) dst_norm.at<float>(j,i) > 150 )
// 						{
// 						 circle( dst_norm_scaled, Point( i, j ), 5,  Scalar(0), 2, 8, 0 );
// 						}
// 				}
// 	 }
//
//  imshow("harris edges",dst_norm_scaled);
//  waitKey(0);

  //blur( src_gray, src_gray, Size(9,9) );
	GaussianBlur( src_gray, src_gray, Size(9,9), 0, 0, BORDER_DEFAULT );

	int kernel = 3;
	int ratio = 3;
	int low_threshold=60;

	Canny(src_gray, edges, low_threshold, low_threshold*ratio, kernel, true);
	#ifdef DEBUG
	imshow("Hough Edges",edges);
	waitKey(0);
	#endif

	vector<Vec4i> lines; //vector holding lines to be detected
	//HoughLinesP(edges, lines, 3, 1*CV_PI/180, 70, 15, 10);
	//HoughLinesP(edges, lines, 1, 2*CV_PI/180, 50, 15, 20);
	HoughLinesP(edges, lines, 1, 2*CV_PI/180, 15, 15, 5);//15,15,5 works

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
	namedWindow("HoughLines",1);
	imshow("HoughLines",src_gray);
	waitKey(0);
	#endif

	// double midThreshold = 0.002;
	double midThreshold = 0.001;//0.0015
	vector<Rect>::iterator it = output.begin();
	cout<<"****Calculating line scores****"<<endl;
	while(it!=output.end()){
		int midScore = 0;
		for(size_t j=0; j<midPoints.size(); j++){
			if((*it).contains(midPoints[j])) midScore++;
		}
		double required = midThreshold * (*it).area();
		cout<<(*it)<<"  \tScore: "<<midScore<<". Required: "<<(int)required<<endl;
		if(midScore<(int)required) it=output.erase(it);
		else ++it;
	}
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

void detectEllipse(vector<EdgePointInfo> edgeList, vector<MyEllipse>& output, int threshold, int minMajor, int maxMajor)
{
	double eps = 0.0001;
	int rotationSpan = 90;
	float minAspectRatio = 0.1;
	size_t edgeNum = edgeList.size();

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
			//distsSq[j][i] = distsSq[i][j];
			if(length<maxMajor*maxMajor && length>minMajor*minMajor)
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

  int randomise = 2;
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

		// vector<int> thirdPointDist_sq;
		// thirdPointDist_sq.resize(edgeNum,0);
		//
		// vector<int> f_sq;
		// f_sq.resize(edgeNum,0);
		//
		// vector<float> costau;
		// costau.resize(edgeNum,0);
		//
		// vector<float> sintau_sq;
		// sintau_sq.resize(edgeNum,0);
		//
		// vector<float> b;
		// b.resize(edgeNum,0);
		//
		// vector<int> bins;
		// bins.resize(edgeNum,0);
		//
		vector<int> accum;
		accum.resize(maxMajor,0);

		int bestMinor = 0;

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
			if(thirdPointDist_sq > a_sq)
			{
				thirdPointDist_sq = -1; //flag that means do not process
			}
			else{
				f_sq = (p.x-x2)*(p.x-x2) + (p.y-y2)*(p.y-y2);
				//cout<<f_sq<<endl;
				//Cosine rule
				costau = ( a_sq + thirdPointDist_sq - f_sq ) / ( 2*sqrt(a_sq*thirdPointDist_sq) );
				costau = min(1.0f, max(-1.0f, costau)); //clip between -1 and 1
				//cout<<costau<<endl;
				sintau_sq = 1-costau*costau;
				b = sqrt( (a_sq * thirdPointDist_sq * sintau_sq) / (a_sq - thirdPointDist_sq * costau * costau + eps) );
				//cout<<b<<endl;
				bin = ceil(b+eps);
				//cout<<bin<<endl;
				local_accum[bin]++;
			}
		}
		#pragma omp critical
		{
			for(int n=0;n<maxMajor;n++){
				accum[n] += local_accum[n];
			}
		}
	  }
		//GaussianBlur( accum, accum, Size(1,5), 0, 0, BORDER_DEFAULT );
		int* element = max_element(&accum[ceil(sqrt(a_sq)*minAspectRatio)],&accum[maxMajor]);
		bestMinor = distance(&accum[0], element);
		if(*element>threshold)
			output.push_back(MyEllipse(x0, y0, atan2(y1-y2,x1-x2), sqrt(a_sq), bestMinor, *element));
		fill(accum.begin(), accum.end(), 0);
	}
	cout<<output.size()<<endl;



	delete(perm); //free memory

}

// void detectEllipse(vector<EdgePointInfo> edgeList, vector<Ellipse>& output, int threshold, int minMajor, int maxMajor, int minMinor, int maxMinor)
// {
// 	int accum[maxMinor] = {0};
// 	for(unsigned int i=0;i<edgeList.size();i++)
// 	{
// 		cout<<i<<endl;
// 		Point p1(edgeList[i].x, edgeList[i].y);
// 		for(unsigned int j=i+1;j<edgeList.size();j++)
// 		{
// 			Point p2(edgeList[j].x, edgeList[j].y);
// 			double dist = norm(p1-p2);
// 			if(dist>=minMajor && dist<=maxMajor)
// 			{
// 				Point p0( (p1+p2)/2 );
// 				double a = dist/2;
// 				double angle = atan2(p2.y-p1.y, p2.x-p1.x);
// 				if(angle>pi/2 || angle<-pi/2) continue;
// 				for(unsigned int k=0;k<edgeList.size();k++)
// 				{
// 					//cout<<k<<endl;
// 					if(k==i||k==j) continue;
// 					Point p(edgeList[k].x, edgeList[k].y);
// 					double d = norm(p-p0);
// 					if(d>minMinor && d < a)
// 					{
// 						double f = norm(p-p2);
// 						double costau = (a*a+d*d-f*f)/(2*a*d);
// 						if(costau<0) continue;
// 						double sintau_sq = 1-costau*costau;
// 						//cout<<"f: "<<f<<" a: "<<a<<" d: "<<d<<" sintau_sq: "<<sintau_sq<<" costau: "<<costau<<endl;
// 						unsigned int b = round( sqrt( (a*a*d*d*sintau_sq)/(a*a-d*d*costau*costau) ) );
// 						//cout<<b<<endl;
// 						accum[b]++;
// 					}
// 				}
// 				int index = 0;
// 				int max = 0;
// 				for(int k=0;k<a;k++)
// 				{
// 					if(accum[k]>max){
// 						max = accum[k];
// 						index = k;
// 					}
// 					accum[k] = 0;
// 				}
// 				if(max>threshold)
// 				{
// 					Ellipse el;
// 					el.xc = p0.x;
// 					el.yc = p0.y;
// 					el.angle = angle;
// 					el.major = dist;
// 					el.minor = 2*index;
// 					cout<<el.xc<<" "<<el.yc<<" "<<el.angle<<" "<<el.major<<" "<<el.minor<<endl;
// 					output.push_back(el);
// 				}
// 			}
// 		}
// 	}
// }

void detectConcentric(vector<EdgePointInfo> edgeList, Size imsize, int min_radius, int max_radius,
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
	//imshow("test",accum);
	//waitKey(0);
}

void extractEdges(Mat& gray_input, vector<Rect> v, vector<EdgePointInfo>& edgeList, int method, int edge_thresh)
{
  float magScale = 255.f/1442.f;
	int scale = 1;
	int delta = 0;
	Mat src = gray_input.clone();
	Mat dy, dx;
	Mat thres, grad;

	thres.create(src.size(),src.type());
	grad.create(src.size(),src.type());

	GaussianBlur( src, src, Size(5,5), 0, 0, BORDER_DEFAULT );

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
