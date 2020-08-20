#ifndef BASTIAN_TRACKING_H
#define BASTIAN_TRACKING_H

#include "GKeyFrame.h"
#include "Map.h"
#include "MarkerFinder.h"
#include "parameters.h"

namespace BASTIAN
{

class GMapPoint;
class GKeyFrame;
class GMap;
class PinholeCamera;

class Tracker
{

enum TRACKING_STATE{
    NOT_INIT = 0,
    TRACKING = 1,
    LOST = 2
};

public:
    Tracker(std::string markerPath, GMap *pGMap_, double fx_, double fy_, double cx_, double cy_, bool bViewer_);
    ~Tracker();

    void ProcessFrame(cv::Mat &image);

    bool GetImageShow(cv::Mat &image);

    PinholeCamera* GetCameraModel(){
        return pPinholeCamera;
    }

    bool GetCurrentPose(Eigen::Matrix4d &Tcw);

private:
    void InitUsingMarkerImage(cv::Mat &gray);

//// optical flow  /////
private:
    std::mutex m_last_frame;
    bool last_frame_is_keyframe;
    cv::Mat last_image;
    GKeyFrame* last_frame;
    bool inBorder(const cv::Point2f &pt);
    void TrackOpticalFlow(cv::Mat &newGray_);
    void UpdateLastFrame(GKeyFrame* new_frame, cv::Mat &new_gray, bool is_keyframe);

/// check whether it is a keyframe ///
private:
    GKeyFrame* last_keyframe; // only used to check new keyframe -> no need to mutex
    double accumulate_parallax_x = 0.0;
    double accumulate_parallax_y = 0.0;
    bool CheckKeyFrame(GKeyFrame* new_frame, int matched_count);

private:
    MarkerFinder *markerFinder;
    GMap *pGMap;

private:
    std::mutex m_state;
    TRACKING_STATE tracking_state;
    PinholeCamera *pPinholeCamera;
    //double fx, fy, cx, cy;

    bool bViewer;

    std::mutex m_image_show;
    cv::Mat image_show;
    cv::Mat image_viewer;

};



}



#endif // BASTIAN_TRACKING_H
