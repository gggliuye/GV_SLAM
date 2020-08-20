#include "MultiMarkerTracker.h"


namespace BASTIAN
{

MultiMarkerTracker::MultiMarkerTracker()
{

}

bool MultiMarkerTracker::AddMarker(const std::string &dataPath, bool view,
              double fx_, double fy_, double cx_, double cy_)
{
    if(marker_count >= MAX_MARKER_NUM)
        return false;
    // initialize the image recoginition
    TrackerTracker* markerFinder = new TrackerTracker(dataPath, view);
    markerFinder->SetCameraParameters(fx_, fy_, cx_, cy_);
    vpTrackerTracker.push_back(markerFinder);
    marker_count++;
    return true;
}

void MultiMarkerTracker::ProcessFrame(cv::Mat &image)
{
    // TODO
}



} // namespace
