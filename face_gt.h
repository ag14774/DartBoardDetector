/////////////////////////////////////////////////////////////////////////////
//
// COMS30121 - face.cpp
//
/////////////////////////////////////////////////////////////////////////////

// header inclusion
// header inclusion
#include <stdio.h>
#include <opencv2/core/core.hpp>
#include <iostream>

using namespace std;
using namespace cv;

String cascade_name = "frontalface.xml";

int sizes[16] = {1,0,0,0,1,11,0,0,0,0,0,0,0,1,2,3};

Rect dart0gt[1] = {Rect(185,205,70,81)};
Rect dart1gt[0];
Rect dart2gt[0];
Rect dart3gt[0];
Rect dart4gt[1] = {Rect(367,108,92,156)};
Rect dart5gt[11] = {Rect(70,133,44,70),Rect(259,175,40,53),Rect(383,199,49,41),Rect(519,181,44,46),
	                  Rect(655,190,44,52),Rect(62,253,48,67),Rect(200,213,46,66),Rect(301,241,39,63),
										Rect(439,239,39,57),Rect(571,244,38,66),Rect(689,224,37,81)};
Rect dart6gt[0];
Rect dart7gt[0];
Rect dart8gt[0];
Rect dart9gt[0];
Rect dart10gt[0];
Rect dart11gt[0];
Rect dart12gt[0];
Rect dart13gt[1] = {Rect(332,123,79,134)};
Rect dart14gt[2] = {Rect(372,210,68,112),Rect(727,192,98,96)};
Rect dart15gt[3] = {Rect(66,137,57,77),Rect(368,111,62,78),Rect(540,126,72,87)};

Rect* positives[16] = {dart0gt,  dart1gt,  dart2gt,  dart3gt,
                       dart4gt,  dart5gt,  dart6gt,  dart7gt,
                       dart8gt,  dart9gt,  dart10gt, dart11gt,
                       dart12gt, dart13gt, dart14gt, dart15gt};

string filenames[16] = {"dart0.jpg", "dart1.jpg",  "dart2.jpg",  "dart3.jpg",
                        "dart4.jpg", "dart5.jpg",  "dart6.jpg",  "dart7.jpg",
                        "dart8.jpg", "dart9.jpg",  "dart10.jpg", "dart11.jpg",
                        "dart12.jpg","dart13.jpg", "dart14.jpg", "dart15.jpg"};

int findIndexOfFile(string filename)
{
	for(int i=0;i<16;i++){
		if(filename.compare(filenames[i]) == 0)
			return i;
	}
	return -1;
}
