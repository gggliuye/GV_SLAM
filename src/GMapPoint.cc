#include "GMapPoint.h"

namespace BASTIAN{

GMapPoint::GMapPoint(Eigen::Vector3d vEigenPose_)
{
    vEigenPose = vEigenPose_;
    bFixed = true;
    {
        std::unique_lock<std::mutex> lock(m_triangulated);
        bTriangulated = true;
    }
    /*{
        std::unique_lock<std::mutex> lock(m_observation);
        vObserveKeyFrames.push_back(pGKeyFrame);
        vIndicesInKeyFrames.push_back(index);
    }*/
    AddUsage();
    AddUsage();
}

GMapPoint::GMapPoint(Eigen::Vector3d vEigenPose_, bool fix)
{
    vEigenPose = vEigenPose_;
    bFixed = fix;
    {
        std::unique_lock<std::mutex> lock(m_triangulated);
        bTriangulated = true;
    }
    /*{
        std::unique_lock<std::mutex> lock(m_observation);
        vObserveKeyFrames.push_back(pGKeyFrame);
        vIndicesInKeyFrames.push_back(index);
    }*/
    AddUsage();
}

GMapPoint::GMapPoint(Eigen::Matrix4d first_camera_, Eigen::Vector2d first_keypoint_)
{
    first_view_set = true;
    first_camera = first_camera_;
    first_keypoint = first_keypoint_;
    AddUsage();
}

GMapPoint::~GMapPoint()
{
}

bool GMapPoint::Triangulate(Eigen::Matrix4d new_camera, Eigen::Vector2d new_keypoint)
{
    if(GetbTriangulated())
        return true;
    if(!first_view_set)
        return false;

    double parallax = abs(new_keypoint(0) - first_keypoint(0)) + abs(new_keypoint(1) - first_keypoint(1));
    //std::cout << parallax << " ";
    if(parallax < 0.15)
        return false;

    // start triangualtion
    Eigen::Matrix4d triangulate_matrix; // = Eigen::Matrix4d::Zero();
    triangulate_matrix.row(0) = first_keypoint(0) * first_camera.row(2) - first_camera.row(0);
    triangulate_matrix.row(1) = first_keypoint(1) * first_camera.row(2) - first_camera.row(1);
    triangulate_matrix.row(2) = new_keypoint(0) * new_camera.row(2) - new_camera.row(0);
    triangulate_matrix.row(3) = new_keypoint(1) * new_camera.row(2) - new_camera.row(1);

    //std::cout << triangulate_matrix << std::endl;
    //Eigen::Vector4d svd_V = Eigen::JacobiSVD<Eigen::Matrix4d>(triangulate_matrix,
    //                            Eigen::ComputeThinV).matrixV().rightCols<1>();
    Eigen::Vector4d svd_V = triangulate_matrix.jacobiSvd(Eigen::ComputeFullV).matrixV().rightCols<1>();
    if(svd_V[3] == 0){
        return false;
    }

    Eigen::Vector4d pose_point;
    pose_point(0) = svd_V(0) / svd_V(3);
    pose_point(1) = svd_V(1) / svd_V(3);
    pose_point(2) = svd_V(2) / svd_V(3);
    pose_point(3) = 1.0;

    // check if point in front of camera
    float depth1 = first_camera.row(2) * pose_point;
    if(depth1 < 0.05){
        return false;
    }

    float depth2 = new_camera.row(2) * pose_point;
    if(depth2 < 0.05){
        return false;
    }

    {
        std::unique_lock<std::mutex> lock(m_pose);
        vEigenPose = pose_point.segment(0,3);
    }

    SetbTriangulated(true);
    return true;
}

/*
bool GMapPoint::TryTriangulationTwoView()
{
    std::cout << " try to triangulate. \n";
    std::unique_lock<std::mutex> lock(m_observation);

    int size_obs = vObserveKeyFrames.size();
    if(size_obs < 2)
        return false;


    // triangluation process
    Eigen::MatrixXd triangulate_matrix(2 * size_obs, 4);

    for(int i  = 0 ; i < size_obs ; i++){
        Eigen::Vector2d observe = vObserveKeyFrames[i]->vTrackedPointsHomo[vIndicesInKeyFrames[i]];
        Eigen::Matrix4d camera_pose = vObserveKeyFrames[i]->GetPoseTcw();
        triangulate_matrix.row(2*i) = observe(0) * camera_pose.row(2) - camera_pose.row(0);
        triangulate_matrix.row(2*i + 1) = observe(1) * camera_pose.row(2) - camera_pose.row(1);
    }
    Eigen::Vector4d svd_V = Eigen::JacobiSVD<Eigen::MatrixXd>(triangulate_matrix,
                                  Eigen::ComputeThinV).matrixV().rightCols<1>();
    {
        std::unique_lock<std::mutex> lock(m_pose);
        vEigenPose(0) = svd_V[0] / svd_V[3];
        vEigenPose(1) = svd_V[1] / svd_V[3];
        vEigenPose(2) = svd_V[2] / svd_V[3];
    }
    //SetbTriangulated(true);

    return false;
}
*/

void GMapPoint::SetBadFlag()
{
    std::unique_lock<std::mutex> lock(m_is_bad);
    bBad = true;
    // clear the vector to save memory
    //vObserveKeyFrames.clear();
}

bool GMapPoint::GetBadFlag()
{
    std::unique_lock<std::mutex> lock(m_is_bad);
    return bBad;
}

void GMapPoint::SetbTriangulated(bool b_in)
{
    std::unique_lock<std::mutex> lock(m_triangulated);
    bTriangulated = b_in;
}

bool GMapPoint::GetbTriangulated()
{
    std::unique_lock<std::mutex> lock(m_triangulated);
    return bTriangulated;
}

Eigen::Vector3d GMapPoint::GetPose()
{
    std::unique_lock<std::mutex> lock(m_pose);
    return vEigenPose;
}

void GMapPoint::SetPose(Eigen::Vector3d vEigenPose_)
{
    std::unique_lock<std::mutex> lock(m_pose);
    vEigenPose = vEigenPose_;
}

void GMapPoint::SetPose(double* pose_array)
{
    std::unique_lock<std::mutex> lock(m_pose);
    vEigenPose << pose_array[0], pose_array[1], pose_array[2];
}

void GMapPoint::AddUsage()
{
    std::unique_lock<std::mutex> lock(m_count);
    count++;
}

// if false -> we can delete this object
bool GMapPoint::SubUsage()
{
    std::unique_lock<std::mutex> lock(m_count);
    return --count;
}

int GMapPoint::GetUsage()
{
    std::unique_lock<std::mutex> lock(m_count);
    return count;
}

} // namespace
