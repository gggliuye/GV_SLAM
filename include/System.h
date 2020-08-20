#ifndef BASTIAN_SYSTEM_H
#define BASTIAN_SYSTEM_H

#include <thread>

#include "Tracking.h"
#include "Viewer.h"

namespace BASTIAN
{

class Tracker;
class GMap;
class GKeyFrame;
class GMapPoint;

class GvSystem
{
public:
    GvSystem(std::string markerPath, bool bView);
    ~GvSystem(){}

    void ProcessFrame(cv::Mat &image);

private:
    Tracker* pTracker;
    GMap* pGMap;
    Viewer* pViewer;

private:
    // threads
    std::thread* pThreadViewer;
    std::thread* pThreadLocalMapping;

private:
    double time_used_one;
    double recent_total_time = 0.0;
    int recent_count = 0;
    int count_interval = 10;
};

} // namespace BASTIAN




#endif //BASTIAN_SYSTEM_H
