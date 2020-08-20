#ifndef MAKER_FINDER_H
#define MAKER_FINDER_H

#include <map>
#include <string>
#include <chrono>
#include <string>
#include <iostream>
#include <fstream>
#include <mutex>

#include <opencv2/core.hpp>
#include <opencv2/features2d.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>

#include "imagecommon.h"
#include "orbextractor.h"

#include "RansacAffine.h"
#include "ImageUtils.h"

namespace BASTIAN
{

class MarkerFinder
{

protected:

enum MarkerTrackingState
{
    NOT_INIT = 0,
    TRACKING = 1,
};

public:

    MarkerFinder(std::string markerfile, bool view);
    ~MarkerFinder();

    void SetCameraParameters(double fx_, double fy_, double cx_, double cy_);

    bool InitSLAM(cv::Mat &gray);

    bool InitSLAM(cv::Mat &gray, std::vector<cv::Point2f> &matchedMarkerPoints,
                                 std::vector<cv::Point3f> &matchedMapPoints,
                                 std::vector<cv::Point2f> &matchedImagePoints, double *pose);

    bool ProcessFrame(cv::Mat &image, cv::Mat &output);

    void GetMatchedImagePointsAndMapPoints(std::vector<cv::Point3f> &matchedMapPoints,
                                           std::vector<cv::Point2f> &matchedImagePoints);

    void GetMarkerInitSize(double &cols, double &rows);

    bool GetGuidePoints(std::vector<cv::Point2f> &imageCorners2D);

////////   init pose estimation using feature points  ////////
protected:
    bool InitializePoseEstimationUsingOrbFeatures(cv::Mat &gray, cv::Mat &homography);

    bool InitializePoseEstimationUsingOrbFeaturesBF(cv::Mat &gray, cv::Mat &homography);

    bool InitializePoseEstimationWithMarkerInCenter(cv::Mat &gray, cv::Mat &homography);

////////   refine homography using image patch     /////////

    bool RefinePoseUsingNCCPatchMatch(cv::Mat &gray, cv::Mat &homography, double ncc_match_threshold,
                                 std::vector<cv::Point2f> &matchedMarkerPoints,
                                 std::vector<cv::Point3f> &matchedMapPoints,
                                 std::vector<cv::Point2f> &matchedImagePoints);

    void CalculatePose(cv::Mat homography, double *pose);

    void CalculatePose(std::vector<cv::Point3f> &matchedMapPoints,
                       std::vector<cv::Point2f> &matchedImagePoints,  double *pose);

protected:
    bool useFeatureInit = false;

    ImageMarkerInfo *markerInfo;
    std::vector<cv::Point3f> vMapPoints;
    std::vector<cv::Point2f> vKeyPoints;

    ORBextractor * mORBFeatureExtractor;

    cv::DescriptorMatcher *mMatcher;
    cv::BFMatcher *matcher_bf;

    bool use_orb_to_refine = false;
    unsigned int min_feature_threshold = 20;
    double ncc_match_threshold_1 = 0.80;
    double ncc_match_threshold_2 = 0.95;
    int ncc_search_radius = 5;
    int ncc_match_patch_width = 17;

    std::mutex mutex_maker_init_size;
    double marker_cols;
    double marker_rows;

protected:
    bool show_viewer = true;

    MarkerTrackingState state;
    double fx, fy, cx, cy;
    cv::Mat mIntrinsic, mDistortionCoeff;

    cv::Mat current_homography;

    bool bHaveInited = false;
    cv::Mat init_homography;

    std::mutex mutex_matched_results;
    std::vector<cv::Point2f> vMatchedMarkerPoints;
    std::vector<cv::Point3f> vMatchedMapPoints;
    std::vector<cv::Point2f> vMatchedImagePoints;


////////  load marker from file ////////////
protected:
    void loadMarker(std::string markerfile);

    //read descriptors
    void readFromBinary(std::ifstream &inputFile, cv::Mat &des);
    //read keypoint
    void readFromBinary(std::ifstream &inputFile, std::vector<cv::KeyPoint> &keyPoint);
    //read A4 3d points
    void read2DPointFromFile(std::ifstream &inputFile, std::vector<cv::Point2f> &p2d);
    //read A4 3d points
    void read3DPointFromFile(std::ifstream &inputFile, std::vector<cv::Point3f> &p3d);

};

} // namespace BASTIAN

#endif //  MAKER_FINDER_H
