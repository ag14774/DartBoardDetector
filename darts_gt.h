/////////////////////////////////////////////////////////////////////////////
//
// COMS30121 - darts.cpp
//
/////////////////////////////////////////////////////////////////////////////

// header inclusion
// header inclusion
#include <stdio.h>
#include <opencv2/core/core.hpp>
#include <iostream>

using namespace std;
using namespace cv;

String cascade_name = "dartcascade/cascade.xml";

Rect dart0gt[1] = {Rect(427,0,193,215)};
Rect dart1gt[1] = {Rect(170,108,248,246)};
Rect dart2gt[2] = {Rect(93,87,108,110),Rect(318,57,67,40)};
Rect dart3gt[1] = {Rect(316,142,83,91)};
Rect dart4gt[1] = {Rect(160,67,205,213)};
Rect dart5gt[1] = {Rect(418,128,134,136)};
Rect dart6gt[1] = {Rect(204,109,76,80)};
Rect dart7gt[1] = {Rect(237,152,165,184)};
Rect dart8gt[2] = {Rect(832,205,141,147),Rect(65,244,68,109)};
Rect dart9gt[2] = {Rect(171,18,296,295),Rect(159,516,66,73)};
Rect dart10gt[3] = {Rect(80,91,117,136),Rect(580,119,65,103),Rect(914,143,39,77)};
Rect dart11gt[2] = {Rect(168,94,73,86),Rect(438,100,38,85)};
Rect dart12gt[1] = {Rect(153,60,71,171)};
Rect dart13gt[1] = {Rect(260,107,159,163)};
Rect dart14gt[2] = {Rect(106,85,156,158),Rect(975,80,151,155)};
Rect dart15gt[1] = {Rect(155,39,148,173)};

int sizes[16] = {1,1,2,1,1,1,1,1,2,2,3,2,1,1,2,1};
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
