#include<iostream>
#include<algorithm>
#include<fstream>
#include<chrono>

#include<opencv2/opencv.hpp>
#include<opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>

#include "imagelocalization.h"

using namespace std;

string dataPath = "/home/yennefer/UTOPA/configurations/maker_images/starORB.dat";
string imagePath = "/home/yennefer/UTOPA/configurations/maker_images/PopArtMarker.jpg";

void makeImageMarkerData(string imageFile, string outputFile)
{
    ImageLocalization imgdll;
    //imgdll.saveImgMarkerInfo(imageFile,0.289,outputFile);
    imgdll.saveImgMarkerInfo(imageFile,5.03,outputFile);
    std::cout << " saved Image Marker Data to " << outputFile << std::endl;
}

/*
int testMain()
{
    cv::VideoCapture cap(0);
    if(!cap.isOpened())
    { 
      cout << "Fail to get camera" << endl; 
      return -1;
    }

    // initialize the image recoginition 
    ImageLocalization imgdll;
    imgdll.Init();
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

	if(!imgdll.ImgMatchingToXml(im,unused, result)){
            std::cout << "  [RESULT] detection failed. " << std::endl << std::endl; 
        }

        cv::imshow("image", imgdll.imageToDraw);
	if(cv::waitKey(30) == 'q'){
            break;
        }
    }
    
    return 0;
}

*/
int main()
{
    makeImageMarkerData(imagePath, 
                 dataPath);
    //testMain();


}

