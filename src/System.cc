#include "System.h"

namespace BASTIAN
{

GvSystem::GvSystem(std::string markerPath, bool bView)
{
    std::cout << "This is the test of GVSLAM. \n";
    std::cout << "Developped by UTOPA, GuangZhou \n";
    std::cout << "This version is made by LIU.YE \n";
    std::cout << "            -- 2020/01 \n\n";

    pGMap = new GMap();
    pTracker = new Tracker(markerPath, pGMap, 635.489, 638.435, 335.128, 218.199, bView);

    if(bView){
        pViewer = new Viewer(pTracker, pGMap, pTracker->GetCameraModel());
        pThreadViewer = new std::thread(&Viewer::Run, pViewer);
    }

    pThreadLocalMapping = new std::thread(&GMap::RunLocalMapping, pGMap);
}

void GvSystem::ProcessFrame(cv::Mat &image)
{
    TicToc tictoc;

    pTracker->ProcessFrame(image);

    // calcualte FPS
    time_used_one = tictoc.toc();
    recent_count++;
    recent_total_time += time_used_one;
    if(recent_count%count_interval == 0){
        double fps = double(count_interval) * 1000 / recent_total_time;
        if(pViewer){
            pViewer->SetFPS(fps);
        }
        recent_count = 0;
        recent_total_time = 0;
    }
}








} // namespace BASTIAN
