#include<iostream>
#include<algorithm>
#include<fstream>
#include<chrono>

#include<opencv2/opencv.hpp>
#include<opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>

#include "imagelocalization.h"

using namespace std;

string dataPath = "/home/yennefer/UTOPA/configurations/maker_images/popORB.dat";
string imagePath = "/home/yennefer/UTOPA/configurations/maker_images/pop.jpg";

void makeImageMarkerData(string imageFile, string outputFile)
{
    ImageLocalization imgdll;
    imgdll.saveImgMarkerInfo(imageFile,0.198,outputFile);
    std::cout << " saved Image Marker Data to " << outputFile << std::endl;
}


int testMain(int video_id)
{
    cv::VideoCapture cap(video_id);
    if(!cap.isOpened())
    { 
      cout << "Fail to get camera" << endl; 
      return -1;
    }

    // initialize the image recoginition 
    ImageLocalization imgdll;

    imgdll.Init(635.489, 638.435, 335.128, 218.199);
    // load the image marker
    imgdll.loadMarker(dataPath);

    // Main loop
    cv::Mat im;
    float result[16];
    string unused = " ";
    while(true)
    {
        cap >> im;

        if(im.empty())
        {
            cerr << endl << "Failed to read image. " << endl;
            break;
        }
        imgdll.imageToDraw = im.clone();

	if(!imgdll.ImgMatchingToXml(im, unused, result)){
            std::cout << "  [RESULT] detection failed. " << std::endl << std::endl; 
        } else {
            std::cout << " detect success \n\n";
        }

        cv::imshow("image", imgdll.imageToDraw);
	if(cv::waitKey(30) == 'q'){
            break;
        }
    }
    
    return 0;
}


int main(int argc, char **argv)
{
    if(argc != 2){
        std::cout << "./test video_id \n";
        return 0;
    }  

    int video_id = atoi(argv[1]);


    makeImageMarkerData(imagePath, 
                 dataPath);
    testMain(video_id);


}

