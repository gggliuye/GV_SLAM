#ifndef MULTI_MARKER_TRACKER_H
#define MULTI_MARKER_TRACKER_H

#include "MarkerTracker.h"

namespace BASTIAN
{


class MultiMarkerTracker
{
public:
    MultiMarkerTracker();
    ~MultiMarkerTracker(){}

    bool AddMarker(const std::string &dataPath, bool view,
                  double fx_, double fy_, double cx_, double cy_);

    void ProcessFrame(cv::Mat &image);

private:
    int marker_count = 0;
    int MAX_MARKER_NUM = 5;
    std::vector<TrackerTracker*> vpTrackerTracker;

};

} // namespace

#endif
