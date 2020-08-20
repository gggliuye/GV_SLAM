#include "GKeyFrame.h"

namespace BASTIAN
{

// Initialize a keyframe with matched marker points
GKeyFrame::GKeyFrame(PinholeCamera *pPinholeCamera_,
                     std::vector<cv::Point3f> &matchedMapPoints,
                     std::vector<cv::Point2f> &matchedImagePoints, bool fix)
{
    pPinholeCamera = pPinholeCamera_;
    //m_tracked_points.lock();
    std::unique_lock<std::mutex> lock(m_map_points);
    size_t point_size = matchedImagePoints.size();
    vTrackedPoints.reserve(point_size);
    vMapPoints.reserve(point_size);
    for(size_t i = 0 ; i < point_size; i++){
        vTrackedPoints.push_back(matchedImagePoints[i]);
        Eigen::Vector3d pose_world(matchedMapPoints[i].x, matchedMapPoints[i].y, matchedMapPoints[i].z);
        GMapPoint* pMapPoint = new GMapPoint(pose_world, fix);
        vMapPoints.push_back(pMapPoint);
    }
    //m_tracked_points.unlock();
/*
    int pts_size = vTrackedPoints.size();
    vTrackedPointsHomo.resize(pts_size);
    for(int i = 0; i < pts_size ; i++){
        vTrackedPointsHomo[i](0) = (vTrackedPoints[i].x - pPinholeCamera->cx) * pPinholeCamera->inv_fx;
        vTrackedPointsHomo[i](1) = (vTrackedPoints[i].y - pPinholeCamera->cy) * pPinholeCamera->inv_fy;
    }
*/
    // todo calculate pose
    mEigenTcw.setIdentity();
    //std::cout << mEigenTcw <<  std::endl;
}

GKeyFrame::GKeyFrame(PinholeCamera *pPinholeCamera_,
          std::vector<GMapPoint*> &newframeMapPints,
          std::vector<cv::Point2f> &matchedImagePoints, bool bFilter)
{
    pPinholeCamera = pPinholeCamera_;
    //m_tracked_points.lock();
    std::unique_lock<std::mutex> lock(m_map_points);

    // only mask point when it is a keyframe
    // to better manager the map points
    if(bFilter){
        num_marker = 0;
        // check if points are too close
        size_t point_size = matchedImagePoints.size();
        vTrackedPoints.reserve(point_size);
        vMapPoints.reserve(point_size);

        cv::Mat mask = cv::Mat(pPinholeCamera->height, pPinholeCamera->width, CV_8UC1, cv::Scalar(255));
        // skip too close points
        for(size_t i = 0 ; i < point_size; i++){
            int row = matchedImagePoints[i].y;
            int col = matchedImagePoints[i].x;
            if(mask.at<uint8_t>(row,col) < 200){
                if(!newframeMapPints[i]->GetbTriangulated()){
                    // delete the point
                    //delete newframeMapPints[i];
                    newframeMapPints[i]->SetBadFlag();
                    newframeMapPints[i] = NULL;
                }
                continue;
            }
            vTrackedPoints.push_back(matchedImagePoints[i]);
            vMapPoints.push_back(newframeMapPints[i]);
            if(newframeMapPints[i] && newframeMapPints[i]->GetFixed()){
                num_marker++;
                cv::circle(mask, matchedImagePoints[i], MIN_DIST_MARKER, 0, -1);
            } else{
                cv::circle(mask, matchedImagePoints[i], MIN_DIST_TRACK, 0, -1);
            }
        }
    } else{
        vTrackedPoints = matchedImagePoints;
        vMapPoints = newframeMapPints;
    }
/*
    int pts_size = vTrackedPoints.size();
    vTrackedPointsHomo.resize(pts_size);
    for(int i = 0; i < pts_size ; i++){
        vTrackedPointsHomo[i](0) = (vTrackedPoints[i].x - pPinholeCamera->cx) * pPinholeCamera->inv_fx;
        vTrackedPointsHomo[i](1) = (vTrackedPoints[i].y - pPinholeCamera->cy) * pPinholeCamera->inv_fy;
    }
*/
    //m_tracked_points.unlock();
    mEigenTcw.setIdentity();
}

GKeyFrame::~GKeyFrame()
{
}

void GKeyFrame::DetectMorePointToTrack(cv::Mat &gray)
{
    Eigen::Matrix4d current_pose = GetPoseTcw();
    int current_point_count = static_cast<int>(vTrackedPoints.size());
    {
        // triangulation points
        for(int i = 0; i < current_point_count; i++){
            GMapPoint* pGMapPoint = vMapPoints[i];
            if(pGMapPoint && !pGMapPoint->GetbTriangulated()){
                Eigen::Vector2d pt_homo;
                pt_homo(0) = (vTrackedPoints[i].x - pPinholeCamera->cx) * pPinholeCamera->inv_fx;
                pt_homo(1) = (vTrackedPoints[i].y - pPinholeCamera->cy) * pPinholeCamera->inv_fy;
                pGMapPoint->Triangulate(current_pose, pt_homo);
            }
        }
    }

    // set mask
    cv::Mat mask = cv::Mat(pPinholeCamera->height, pPinholeCamera->width, CV_8UC1, cv::Scalar(255));
    for (int i = 0; i < current_point_count; i++){
        cv::circle(mask, vTrackedPoints[i], MIN_DIST, 0, -1);
    }
    std::vector<cv::Point2f> new_points;
    //std::cout << num_marker << "\n";
    int num_to_detect = MAX_CNT - current_point_count + num_marker;
    if( num_to_detect > 0 ){
        cv::goodFeaturesToTrack(gray, new_points, num_to_detect, 0.01, MIN_DIST, mask);

        std::unique_lock<std::mutex> lock(m_map_points);
        for(size_t i = 0 ; i < new_points.size(); i ++){
            vTrackedPoints.push_back(new_points[i]);

            Eigen::Vector2d pt_homo;
            pt_homo(0) = (new_points[i].x - pPinholeCamera->cx) * pPinholeCamera->inv_fx;
            pt_homo(1) = (new_points[i].y - pPinholeCamera->cy) * pPinholeCamera->inv_fy;

            // add new map point
            GMapPoint* pGMapPoint = new GMapPoint(current_pose, pt_homo);
            vMapPoints.push_back(pGMapPoint);
            //vMapPoints.push_back(NULL);
        }
    }
    //std::cout << vTrackedPointsHomo.size() << " " << vTrackedPoints.size() << std::endl;
}


std::vector<GMapPoint*> GKeyFrame::GetMapPoints()
{
    std::unique_lock<std::mutex> lock(m_map_points);
    //return vMapPoints;
    return std::vector<GMapPoint*>(vMapPoints.begin(), vMapPoints.end());
}

// more calculations here (mostly the inverse operation of a transformation matrix)
// but we donnot care, as it is only used for debug viewer
cv::Mat GKeyFrame::GetPoseTwc()
{
    std::unique_lock<std::mutex> lock(m_pose);
    Eigen::Matrix4d mEigenTwc;
    mEigenTwc.setIdentity();

    Eigen::Matrix3d rotation_matrix = mEigenTcw.block(0,0,3,3);
    Eigen::Vector3d transpose_vector(mEigenTcw(0,3), mEigenTcw(1,3), mEigenTcw(2,3));
    mEigenTwc.block(0,0,3,3) = rotation_matrix.transpose();

    Eigen::Vector3d inverse_transpose = - rotation_matrix.transpose() * transpose_vector;
    mEigenTwc(0,3) = inverse_transpose(0);
    mEigenTwc(1,3) = inverse_transpose(1);
    mEigenTwc(2,3) = inverse_transpose(2);

    cv::Mat cvMat(4,4,CV_32F);
    for(int i = 0; i < 4 ; i++)
        for(int j = 0; j < 4 ; j++)
            cvMat.at<float>(i,j) = mEigenTwc(i,j);
    //std::cout << cvMat <<  std::endl;
    return cvMat.clone();
}

Eigen::Matrix4d GKeyFrame::GetPoseTcw()
{
    std::unique_lock<std::mutex> lock(m_pose);
    return mEigenTcw;
}

Eigen::Vector3d GKeyFrame::GetWorldCoord()
{
    Eigen::Vector3d res;
    std::unique_lock<std::mutex> lock(m_pose);
    res << mEigenTcw(0,3), mEigenTcw(1,3), mEigenTcw(2,3);
    return res;
}

void GKeyFrame::GetPoseTcwArray(double *pose)
{
    std::unique_lock<std::mutex> lock(m_pose);
    Eigen::Matrix3d rotation_matrix = mEigenTcw.block(0,0,3,3);
    Eigen::Quaterniond qe(rotation_matrix);
    pose[0] = mEigenTcw(0,3); pose[1] = mEigenTcw(1,3); pose[2] = mEigenTcw(2,3);
    pose[3] = qe.x(); pose[4] = qe.y(); pose[5] = qe.z(); pose[6] = qe.w();
}

void GKeyFrame::SetPoseTcwArray(double *pose)
{
    std::unique_lock<std::mutex> lock(m_pose);
    mEigenTcw.setIdentity();
    Eigen::Quaterniond qc(pose[6], pose[3], pose[4], pose[5]);
    mEigenTcw.block(0,0,3,3) = qc.matrix();
    mEigenTcw(0,3) = pose[0]; mEigenTcw(1,3) = pose[1]; mEigenTcw(2,3) = pose[2];
}

}// namespace BASTIAN
