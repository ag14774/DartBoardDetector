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
//#include "face_gt.h"
#include "darts_gt.h"

#define MAX_CONCENTRIC 20
#define EDGEDETECT 1 //1 for CannyEdge

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

/** Function Headers */
void detect( Mat& frame, vector<Rect>& output );
void drawRects(Mat& frame, Mat& output, vector<Rect> v);
                                                                           //inverse of resolution in X, Y and R direction
void detectConcentric(vector<EdgePointInfo> edgeList, Size imsize, int min_radius, int max_radius,
											int threshold, int resX, int resY, int resR, vector<ConcentricCircles>& output);
void extractEdges(Mat& gray_input, vector<EdgePointInfo>& edgeList, int edge_thresh);
double rectIntersection(Rect A, Rect B);
double fscore(vector<Rect> ground, vector<Rect> detected);

/** Global variables */
CascadeClassifier cascade;

/** @function main */
int main( int argc, const char** argv )
{
  // 1. Read Input Image
	Mat frame = imread(argv[1], CV_LOAD_IMAGE_COLOR);
  Mat src_gray;

	/// Convert it to gray
  cvtColor( frame, src_gray, COLOR_BGR2GRAY );

	string fullname(argv[1]);
	int index = findIndexOfFile(fullname);
	//string noext = fullname.substr(0, fullname.find_last_of("."));

  vector<Rect> output;
	Mat frame_output;

	// 2. Load the Strong Classifier in a structure called `Cascade'
	if( !cascade.load( cascade_name ) ){ printf("--(!)Error loading\n"); return -1; };

	// 3. Detect dartboards
	detect( frame, output );

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

	// 1. Prepare Image by turning it into Grayscale
	cvtColor( frame, frame_gray, CV_BGR2GRAY );


//***************TESTING*********************
  vector<EdgePointInfo> edges;
	extractEdges(frame_gray, edges, 25);
	cout<<edges.size()<<endl;
	vector<ConcentricCircles> circs;
	int min_radius=15,max_radius=250,thres=350,resX=5,resY=5,resR=5;
	detectConcentric(edges, frame_gray.size(), min_radius, max_radius, thres, resX, resY, resR, circs);
	cout<<circs.size()<<endl;
	for(vector<ConcentricCircles>::iterator it = circs.begin();it!=circs.end();++it){
		cout << (*it).xc <<" "<<(*it).yc<<" ";
		for(int i=0;i<(*it).num;i++){
			cout << (*it).rs[i] << " ";
		}
		cout << endl;
	}
//*******************************************

  //Normalise lighting
  equalizeHist( frame_gray, frame_gray );

	// 2. Perform Viola-Jones Object Detection
	cascade.detectMultiScale( frame_gray, output, 1.1, 1, 0|CV_HAAR_SCALE_IMAGE, Size(50, 50), Size(500,500) );

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

bool valid(int num, int min, int max){
  if(num<min)
    return false;
  if(num>=max)
    return false;
  return true;
}

void detectConcentric(vector<EdgePointInfo> edgeList, Size imsize, int min_radius, int max_radius,
											int threshold, int resX, int resY, int resR, vector<ConcentricCircles>& output)
{
	int min_distance = max(imsize.height,imsize.width)/8;
	cout<<min_distance<<endl;
	int vote_around_radius = 0; //measured in bins
	int SHIFT = 10; int ONE = 1<<SHIFT;
  int ndims = 3;
	double low_threshold = threshold * 0.2;
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

void extractEdges(Mat& gray_input, vector<EdgePointInfo>& edgeList, int edge_thresh)
{
  float magScale = 255.f/1442.f;
	int scale = 1;
	int delta = 0;
	Mat src = gray_input.clone();
	Mat dy, dx;
	Mat thres, grad;

	thres.create(src.size(),src.type());
	grad.create(src.size(),src.type());

	GaussianBlur( src, src, Size(7,7), 0, 0, BORDER_DEFAULT );

  int lowThreshold = 50;
	int ratio = 3;
	int kernel_size = 3;
	Canny( src,  thres, lowThreshold, lowThreshold*ratio, kernel_size );

	Sobel( src, dx, CV_16S, 1, 0, 3, scale, delta, BORDER_DEFAULT );
	Sobel( src, dy, CV_16S, 0, 1, 3, scale, delta, BORDER_DEFAULT );

  int nRows = src.rows;
	int nCols = src.cols;

  for(int i=0;i<nRows;++i)
	{
		short int* dx_ptr = dx.ptr<short int>(i);
		short int* dy_ptr = dy.ptr<short int>(i);
		uchar* thres_ptr  = thres.ptr<uchar>(i);
		uchar* grad_ptr    = grad.ptr<uchar>(i);
    for(int j = 0;j<nCols;j++)
		{
			//TODO: CHECK IF POINT IS CONTAINED IN ONE OF THE RECTS RETURNED BY VIOLA
			double psi = atan2((double)dy_ptr[j]/8.0,(double)dx_ptr[j]/8.0);
			grad_ptr[j] = (uchar)255*(psi-(-pi))/(2*pi);
			#if EDGEDETECT==0
			double mag = magScale*sqrt(dx_ptr[j]*dx_ptr[j]+dy_ptr[j]*dy_ptr[j]);
			if(mag > edge_thresh){
				thres_ptr[j] = 255;
				EdgePointInfo epi;
				epi.x = j;
				epi.y = i;
				epi.grad = psi;
				edgeList.push_back(epi);
			}
			else{
				thres_ptr[j] = 0;
			}
			#else
			if(thres_ptr[j]==255){
				EdgePointInfo epi;
				epi.x = j;
				epi.y = i;
				epi.grad = psi;
				edgeList.push_back(epi);
			}
			#endif
		}
	}

	namedWindow("Gradient",CV_WINDOW_AUTOSIZE);
	imshow("Gradient", grad);
	namedWindow("Thresholded magnitude",CV_WINDOW_AUTOSIZE);
	imshow("Thresholded magnitude", thres);
	waitKey(0);

}
