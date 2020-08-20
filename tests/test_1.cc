#include <iostream>

#include<opencv2/opencv.hpp>
#include<opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>

#include "GKeyFrame.h"
#include "MarkerFinder.h"
#include "Test_Cpp.h"
#include "System.h"

using namespace BASTIAN;
using namespace std;

string dataPath = "/home/yennefer/UTOPA/configurations/maker_images/popORB.dat";

int testMain(int video_id);

int main(int argc, char **argv)
{
    if(argc != 2){
        std::cout << "cpulimit -l 100 ./test_1 video_id \n";
        return 0;
    }

    //BASTIAN_TEST::TestOptimization();

    int video_id = atoi(argv[1]);
    testMain(video_id);
    return 0;
}

int testMain(int video_id)
{
    GvSystem *pGvSystem = new GvSystem(dataPath, true);
    cv::VideoCapture cap(video_id);
    if(!cap.isOpened()){
      cout << "Fail to get camera" << endl;
      return -1;
    }

    // Main loop
    cv::Mat im;
    while(true){
        cap >> im;
        if(im.empty()){
            cerr << endl << "Failed to read image. " << endl;
            break;
        }
        pGvSystem->ProcessFrame(im);
    }

    return 0;
}
