#ifndef MARKER_TRACKER_H
#define MARKER_TRACKER_H

#include<opencv2/core/core.hpp>
#include<opencv2/opencv.hpp>

#include "MarkerFinder.h"
#include "BruteFinder.h"

namespace BASTIAN
{

class TrackerTracker : public MarkerFinder
{

public:
    TrackerTracker(std::string markerfile, bool view);
    ~TrackerTracker(){}

    bool ProcessFrame(cv::Mat &image, cv::Mat &output);

private:
    int BORDER_SIZE = 3;
    int WIDTH = 640;
    int HEIGHT = 480;
    bool inBorder(const cv::Point2f &pt);
    cv::Mat TrackingUseOpticalFlow(cv::Mat &gray);

    cv::Mat last_gray;
    std::vector<cv::Point2f> last_maker_points;
    std::vector<cv::Point2f> last_image_points;
    //vMatchedMarkerPoints, vMatchedMapPoints, vMatchedImagePoints

private:

    /// fast match
    int fast_match_patch_size = 10;
    int fast_match_search_size = 2;
    double fast_ncc_match_threshold = 0.6;
    int fast_interval = 2;
    bool FastRefinePoseUsingNCCPatchMatch(cv::Mat &gray, cv::Mat &homography,
                                    std::vector<cv::Point2f> &matchedMarkerPoints,
                                    std::vector<cv::Point2f> &matchedImagePoints);

    /// init match
    int init_match_patch_size = 20;
    int init_match_search_size = 2;
    double init_ncc_match_threshold = 0.7;
    bool InitPosePatchBrute(cv::Mat &gray, cv::Mat &homography, cv::Mat &output);


    int LOST_PTS_NUM = 10;


    BruteFinder* pBruteFinder;
};


} // namespace
#endif
