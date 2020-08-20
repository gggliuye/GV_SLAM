#ifndef G_MAP_POINT_H
#define G_MAP_POINT_H

#include "utils.hpp"
#include "GKeyFrame.h"

namespace BASTIAN
{

class GKeyFrame;

class GMapPoint{

public:
    // set using marker point
    GMapPoint(Eigen::Vector3d vEigenPose_);
    GMapPoint(Eigen::Vector3d vEigenPose_, bool fix);
    // init normal
    GMapPoint(Eigen::Matrix4d first_camera_, Eigen::Vector2d first_keypoint_);
    ~GMapPoint();

    void SetBadFlag();
    bool GetBadFlag();

    bool GetFixed(){
        return bFixed;
    }

    void SetbTriangulated(bool b_in);
    bool GetbTriangulated();

    Eigen::Vector3d GetPose();
    void SetPose(Eigen::Vector3d vEigenPose_);
    void SetPose(double* pose_array);

///////  for triangluation  /////////////
public:
    bool first_view_set = false;
    Eigen::Matrix4d first_camera;
    Eigen::Vector2d first_keypoint;
    bool Triangulate(Eigen::Matrix4d new_camera, Eigen::Vector2d new_keypoint);
    //bool TryTriangulation();
    //bool TryTriangulationTwoView();

private:
    /// these are only used for triangluation ///
    //std::mutex m_observation;
    //std::vector<GKeyFrame*> vObserveKeyFrames;
    //std::vector<int> vIndicesInKeyFrames;


public:

    /// world pose of the point ////
    std::mutex m_pose;
    Eigen::Vector3d vEigenPose;

    // flag for bad point
    std::mutex m_is_bad;
    bool bBad = false;

    // whether it is triangulated
    std::mutex m_triangulated;
    bool bTriangulated = false;

public:
    /////// parameters for optimization  //////
    // id of local optimzaton parameter block
    int local_parameter_id = -1;
    double pose_local_opt[3];
    int frame_parameter_id = -1;
    double pose_frame_opt[3];

private:
    // if truth : when it is a marker image point -> is a prior
    // if false : normal point
    bool bFixed = false;

private:
    //////// parameter for loop detection //////
    cv::Mat mCvDes;


    /////// flag to delete the pointer //////////
    // these should only be used in tracking thread ///
    // once the map point added to the map, we won't delete it //
private:
    std::mutex m_count;
    int count = 0;
public:
    void AddUsage();
    // if false -> we can delete this object
    bool SubUsage();
    int GetUsage();

};

}

#endif
