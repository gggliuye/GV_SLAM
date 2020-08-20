#include "Tracking.h"

namespace BASTIAN
{

Tracker::Tracker(std::string markerPath, GMap *pGMap_, double fx_, double fy_, double cx_, double cy_, bool bViewer_):
             pGMap(pGMap_), bViewer(bViewer_)
{
    markerFinder = new MarkerFinder(markerPath, false);
    markerFinder->SetCameraParameters(fx_, fy_, cx_, cy_);

    pPinholeCamera = new PinholeCamera(fx_, fy_, cx_, cy_, 640, 480);

    tracking_state = NOT_INIT;
}

Tracker::~Tracker()
{

}

void Tracker::ProcessFrame(cv::Mat &image)
{
    if(bViewer){
        image_show = image.clone();
    }
    cv::Mat imageGray;
    if (image.channels() == 3) {
        cv::cvtColor(image, imageGray, CV_RGB2GRAY);
    } else if (image.channels() == 1){
        imageGray = image.clone();
    }

    if(tracking_state == NOT_INIT){
        InitUsingMarkerImage(imageGray);
    } else if(tracking_state == TRACKING){
        TrackOpticalFlow(imageGray);
    } else if(tracking_state == LOST){
        InitUsingMarkerImage(imageGray);
    }

}

void Tracker::InitUsingMarkerImage(cv::Mat &gray)
{
    std::vector<cv::Point2f> matchedMarkerPoints;
    std::vector<cv::Point3f> matchedMapPoints;
    std::vector<cv::Point2f> matchedImagePoints;
    double pose[7];
    if(markerFinder->InitSLAM(gray, matchedMarkerPoints,
            matchedMapPoints, matchedImagePoints, pose))
    {
        if(matchedMarkerPoints.size() > 100){
            GKeyFrame* gKF = new GKeyFrame(pPinholeCamera, matchedMapPoints,
                                        matchedImagePoints);
            gKF->SetPoseTcwArray(pose);
            pGMap->ReInitMapWithMarkerPoints(gKF);
            gKF->DetectMorePointToTrack(gray);
            last_image = gray;
            last_keyframe = gKF;
            {
                std::unique_lock<std::mutex> lock(m_last_frame);
                last_frame = gKF;
                last_frame_is_keyframe = true;
            }
            {
                std::unique_lock<std::mutex> lock(m_state);
                tracking_state = TRACKING;
            }
        } else {
            {
                std::unique_lock<std::mutex> lock(m_state);
                tracking_state = NOT_INIT;
            }
        }
    } else {
        {
            std::unique_lock<std::mutex> lock(m_state);
            tracking_state = NOT_INIT;
        }
    }

    // draw output image
    if(bViewer && ! image_show.empty()){
        cv::Point pt_text = cv::Point(5,  image_show.rows - 10);
        if(tracking_state == NOT_INIT){
            // Show block to guide user
            std::vector<cv::Point2f> imageCorners2D;
            markerFinder->GetGuidePoints(imageCorners2D);
            cv::line(image_show, imageCorners2D[0], imageCorners2D[1], cv::Scalar(255, 0, 0), 2 );
            cv::line(image_show, imageCorners2D[1], imageCorners2D[2], cv::Scalar(255, 0, 0), 2 );
            cv::line(image_show, imageCorners2D[2], imageCorners2D[3], cv::Scalar(255, 0, 0), 2 );
            cv::line(image_show, imageCorners2D[3], imageCorners2D[0], cv::Scalar(255, 0, 0), 2 );
            cv::putText(image_show, "detect fail", pt_text, cv::FONT_HERSHEY_TRIPLEX, 0.8, cv::Scalar(255, 200, 200), 2, CV_AA);
        } else if(tracking_state == TRACKING){
            for(size_t i = 0 ; i < matchedImagePoints.size(); i ++){
                cv::circle(image_show, matchedImagePoints[i], 5, cv::Scalar(0, 0, 255), 2);
            }
            cv::putText(image_show, "detect success", pt_text, cv::FONT_HERSHEY_TRIPLEX, 0.8, cv::Scalar(255, 200, 200), 2, CV_AA);
        }
        std::unique_lock<std::mutex> lock(m_image_show);
        image_viewer = image_show.clone();
    }
}

bool Tracker::inBorder(const cv::Point2f &pt)
{
    return BORDER_SIZE <= pt.x && pt.x < pPinholeCamera->width - BORDER_SIZE
    && BORDER_SIZE <= pt.y && pt.y < pPinholeCamera->height - BORDER_SIZE;
}

void Tracker::TrackOpticalFlow(cv::Mat &newGray_)
{
    cv::Mat newGray;
    cv::Ptr<cv::CLAHE> clahe = cv::createCLAHE(3.0, cv::Size(8, 8));
    clahe->apply(newGray_, newGray);

    std::vector<GMapPoint*> keyframeMapPints = last_frame->GetMapPoints();

    // points for this frame
    std::vector<cv::Point2f> new_optical_pts;
    std::vector<GMapPoint*> newframeMapPints;
    newframeMapPints.reserve(keyframeMapPints.size());

    std::vector<uchar> status;
    std::vector<float> err;
    cv::calcOpticalFlowPyrLK(last_image, newGray, last_frame->vTrackedPoints, new_optical_pts, status, err, cv::Size(31, 31), 2);

    int j = 0;
    std::vector<cv::Point2f> last_optical_pts;
    {
        for (unsigned int i = 0; i < status.size(); i++){
            if (status[i] && inBorder(new_optical_pts[i])){
                new_optical_pts[j++] = new_optical_pts[i];
                newframeMapPints.push_back(keyframeMapPints[i]);
                last_optical_pts.push_back(last_frame->vTrackedPoints[i]);
            }
        }
        new_optical_pts.resize(j);
    }

    // reject match with fundamental matrix
    std::vector<uchar> status_fund;
    cv::findFundamentalMat(last_optical_pts, new_optical_pts, cv::FM_RANSAC, F_THRESHOLD, 0.99, status_fund);
    j = 0;
    double current_parallax_x = 0.0;
    double current_parallax_y = 0.0;
    int f_count = 0;
    {
        for (unsigned int i = 0; i < status_fund.size(); i++){
            if (status_fund[i]){
                current_parallax_x += last_optical_pts[i].x - new_optical_pts[i].x;
                current_parallax_y += last_optical_pts[i].y - new_optical_pts[i].y;
                newframeMapPints[j] = newframeMapPints[i];
                new_optical_pts[j++] = new_optical_pts[i];
            } else {
                f_count++;
            }
        }
        accumulate_parallax_x += current_parallax_x / j;
        accumulate_parallax_y += current_parallax_y / j;
        new_optical_pts.resize(j);
    }
    //std::cout << " fundamental outlier : " << f_count << ".\n";


    GKeyFrame* thisFrame = new GKeyFrame(pPinholeCamera, newframeMapPints, new_optical_pts, true);
    double pose[7];
    last_frame->GetPoseTcwArray(pose);
    thisFrame->SetPoseTcwArray(pose);

    //int matched_count = CERES_OPTIMIZATION::OptimizeCameraPoseAndMapPoint(thisFrame);
    int matched_count = CERES_OPTIMIZATION::OptimizeCameraPose(thisFrame);
    if(matched_count < MIN_TRACKED_PTS){
        {
            std::unique_lock<std::mutex> lock(m_state);
            tracking_state = LOST;
        }
        return;
    }

    if(CheckKeyFrame(thisFrame, matched_count)){
        thisFrame->DetectMorePointToTrack(newGray);
        UpdateLastFrame(thisFrame, newGray, true);
        pGMap->AddNewKeyFrame(thisFrame);
    } else{
        UpdateLastFrame(thisFrame, newGray, false);
    }

    // draw output image
    if(bViewer && !image_show.empty()){
        cv::Point pt_text = cv::Point(5, image_show.rows - 10);
        for(size_t i = 0 ; i < new_optical_pts.size(); i ++){
            if(newframeMapPints[i]){
                if(newframeMapPints[i]->GetbTriangulated()){
                    cv::circle(image_show, new_optical_pts[i], 5, cv::Scalar(0, 0, 255), 2);
                } else {
                    cv::circle(image_show, new_optical_pts[i], 5, cv::Scalar(0, 255, 0), 2);
                }
            } else {
                cv::circle(image_show, new_optical_pts[i], 5, cv::Scalar(0, 255, 0), 2);
            }
        }
        cv::putText(image_show, "Tracking", pt_text, cv::FONT_HERSHEY_TRIPLEX, 0.8, cv::Scalar(255, 200, 200), 2, CV_AA);

        std::unique_lock<std::mutex> lock(m_image_show);
        image_viewer = image_show.clone();
    }
}

void Tracker::UpdateLastFrame(GKeyFrame* new_frame, cv::Mat &new_gray, bool is_keyframe)
{
    {
        std::unique_lock<std::mutex> lock(m_last_frame);
        if(!last_frame_is_keyframe){
            GKeyFrame* to_delete = last_frame;
            delete to_delete;
        }
        last_frame = new_frame;
        last_frame_is_keyframe = is_keyframe;
    }
    last_image = new_gray;
}

bool Tracker::GetCurrentPose(Eigen::Matrix4d &Tcw)
{
    std::unique_lock<std::mutex> lock1(m_state);
    std::unique_lock<std::mutex> lock2(m_last_frame);
    if(tracking_state == TRACKING){
        Tcw = last_frame->GetPoseTcw();
        return true;
    } else{
        return false;
    }
}

bool Tracker::GetImageShow(cv::Mat &image)
{
    if(bViewer && !image_viewer.empty()){
        std::unique_lock<std::mutex> lock(m_image_show);
        image = image_viewer.clone();
        return true;
    } else {
        return false;
    }
}

bool Tracker::CheckKeyFrame(GKeyFrame* new_frame, int matched_count)
{
    bool add = false;
    double accumulate_parallax = abs(accumulate_parallax_x) + abs(accumulate_parallax_y);
    if(accumulate_parallax > KEYFRAME_PARALLAX){
        add = true;
    } else if(matched_count < KEYFRAME_TRACKED_POINTS){
        add = true;
    }

    if(add){
        last_keyframe = new_frame;
        accumulate_parallax_x = 0.0;
        accumulate_parallax_y = 0.0;
        return true;
    } else {
        return false;
    }
}

} // namespace
