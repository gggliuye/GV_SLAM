#include <chrono>

#include<opencv2/opencv.hpp>
#include<opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>

#include "MarkerTracker.h"

using namespace std;

string dataPath = "/home/yennefer/UTOPA/configurations/maker_images/popORB.dat";

class TicToc
{
  public:
    TicToc()
    {
        tic();
    }

    void tic()
    {
        start = std::chrono::system_clock::now();
    }

    double toc()
    {
        end = std::chrono::system_clock::now();
        std::chrono::duration<double> elapsed_seconds = end - start;
        return elapsed_seconds.count() * 1000;
    }

  private:
    std::chrono::time_point<std::chrono::system_clock> start, end;
};

int testMain(int video_id)
{
    // initialize the image recoginition
    BASTIAN::TrackerTracker markerFinder(dataPath, true);
    markerFinder.SetCameraParameters(635.489, 638.435, 335.128, 218.199);

    cv::VideoCapture cap(video_id);
    if(!cap.isOpened())
    {
      cout << "Fail to get camera" << endl;
      return -1;
    }

    // PPS calculation
    double time_used_one;
    double recent_total_time = 0.0;
    int recent_count = 0;
    int count_interval = 10;
    int fps = 0;

    // Main loop
    cv::Mat im, im_show;
    while(true)
    {
        TicToc tictoc;
        cap >> im;

        if(im.empty())
        {
            cerr << endl << "Failed to read image. " << endl;
            break;
        }

        cv::Point pt_text = cv::Point(5,  im.rows - 10);
	    if(!markerFinder.ProcessFrame(im, im_show)){
            cv::putText(im_show, "detect fail", pt_text, cv::FONT_HERSHEY_TRIPLEX, 0.8, cv::Scalar(255, 200, 200), 2, CV_AA);
            //std::cout << " [RESULT] detect fail. " << std::endl;
        } else {
            cv::putText(im_show, "detect success", pt_text, cv::FONT_HERSHEY_TRIPLEX, 0.8, cv::Scalar(255, 200, 200), 2, CV_AA);
            //std::cout << " detect success \n\n";
        }

        // calcualte FPS
        time_used_one = tictoc.toc();
        recent_count++;
        recent_total_time += time_used_one;
        if(recent_count%count_interval == 0){
            fps = double(count_interval) * 1000 / recent_total_time;
            recent_count = 0;
            recent_total_time = 0;
        }
        pt_text = cv::Point(5, 20);
        cv::putText(im_show, "FPS:"+std::to_string(fps), pt_text, cv::FONT_HERSHEY_SIMPLEX, 0.8, cv::Scalar(255, 50, 50), 2, CV_AA);

        cv::imshow("image", im_show);
	    if(cv::waitKey(30) == 'q'){
            break;
        }

    }

    return 0;
}


int main(int argc, char **argv)
{
    if(argc != 2){
        std::cout << "cpulimit -l 100 ./test_ly video_id \n";
        return 0;
    }

    int video_id = atoi(argv[1]);

    testMain(video_id);


}
