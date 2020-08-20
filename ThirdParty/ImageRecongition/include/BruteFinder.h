#ifndef BRUTE_FINDER_H
#define BRUTE_FINDER_H

#include <vector>
#include <iostream>

#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/opencv.hpp>

namespace BASTIAN
{

class BruteFinder {

public:
    BruteFinder(cv::Mat &imageMarker_, std::vector<cv::Point2f> vMarkerCorners2D_, bool view);
    ~BruteFinder(){}

    bool ProcessGray(cv::Mat &gray, cv::Mat &homography, cv::Mat &show_image);

private:
    float NccCalculation(cv::Mat &image, int level, cv::Mat &show_image, cv::Point2f &best_point);
    float NccOnce(cv::Mat &image, cv::Mat &marker, int col0, int row0,
        int patchSize, float patchSum, float patchSqSum, float patchMean);

    void DrawRectangle(cv::Mat &output, std::vector<cv::Point2f> &imageCorners2D, int level, float best_score);
private:
    bool bView = false;
    cv::Mat imageMarker;
    //markerInfo->markerCorners2D
    std::vector<cv::Point2f> vMarkerCorners2D;

private:
    int COLS = 640, ROWS = 480;
    int gaussian_kernel = 3;
    float rescale_ratio = 0.3;
    int base_interval = 2;

    int nLevel = 16;
    double dScale = 0.8;
    float NCC_THRESHOLD = 0.2;
    std::vector<double> vScales;
    std::vector<double> vScalesInv;
    std::vector<cv::Mat> vMarkerPyramid;

    std::vector<int> vMarkerPatchSize;
    std::vector<float> vMarkerPatchSum;
    std::vector<float> vMarkerPatchSqSum;
    std::vector<float> vMarkerPatchMean;

};


} // namespace

#endif
