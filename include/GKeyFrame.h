#ifndef G_KEY_FRAME_H
#define G_KEY_FRAME_H

#include "utils.hpp"
#include "GMapPoint.h"
#include "Optimization.h"
#include "parameters.h"

namespace BASTIAN
{

class GMapPoint;

class PinholeCamera
{
public:
    PinholeCamera(double fx_, double fy_, double cx_, double cy_, int width_, int height_):
        fx(fx_), fy(fy_), cx(cx_), cy(cy_),width(width_), height(height_){
            inv_fx = 1.0/fx;
            inv_fy = 1.0/fy;
        }

    double fx;
    double fy;
    double cx;
    double cy;
    int width;
    int height;
    double inv_fx;
    double inv_fy;
};

class GKeyFrame{
public:
    GKeyFrame(PinholeCamera *pPinholeCamera_,
              std::vector<cv::Point3f> &matchedMapPoints,
              std::vector<cv::Point2f> &matchedImagePoints, bool fix = true);

    GKeyFrame(PinholeCamera *pPinholeCamera_,
              std::vector<GMapPoint*> &newframeMapPints,
              std::vector<cv::Point2f> &matchedImagePoints, bool bFilter);
    ~GKeyFrame();

    std::vector<GMapPoint*> GetMapPoints();


/////////// for optical flow tracking /////////
public:
    void DetectMorePointToTrack(cv::Mat &gray);


/////////// for the pose /////////
public:
    cv::Mat GetPoseTwc();
    Eigen::Matrix4d GetPoseTcw();
    Eigen::Vector3d GetWorldCoord();
    // get and set pose for optimization
    // pose is : x, y, z, qx, qy, qz, qw
    void GetPoseTcwArray(double *pose);
    void SetPoseTcwArray(double *pose);


public:
    PinholeCamera *pPinholeCamera;
    // the tracked key points
    //std::vector<cv::KeyPoint> vKeyPoints;

    // optical flow points
    // std::mutex m_tracked_points;
    // not change its values, once initialized the keyframe -> not using mutex
    std::vector<cv::Point2f> vTrackedPoints;
    //std::vector<Eigen::Vector2d> vTrackedPointsHomo;

private:
    int num_marker = 0;

    // the corresponding map points, if NULL then it has no correspondence
    std::mutex m_map_points;
    std::vector<GMapPoint*> vMapPoints;

    std::mutex m_pose;
    //Eigen::Matrix4d mEigenTwc;  // used in viewer
    Eigen::Matrix4d mEigenTcw;  // used in optimization
};



} // namespace BASTIAN


#endif //  # ifndef  G_KEY_FRAME_H
