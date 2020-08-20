#include "MarkerTracker.h"

namespace BASTIAN
{

TrackerTracker::TrackerTracker(std::string markerfile, bool view) : MarkerFinder(markerfile, view)
{
    pBruteFinder = new BruteFinder(markerInfo->markerImage, markerInfo->markerCorners2D, false);
    state = NOT_INIT;
}

bool TrackerTracker::ProcessFrame(cv::Mat &image, cv::Mat &output)
{
    output = image.clone();
    cv::Mat imageGray_;
    if (image.channels() == 3) {
        cv::cvtColor(image, imageGray_, CV_RGB2GRAY);
    } else if (image.channels() == 1){
        imageGray_ = image;
    }

    // equalize the image
    cv::Mat imageGray;
    cv::Ptr<cv::CLAHE> clahe = cv::createCLAHE(3.0, cv::Size(8, 8));
    clahe->apply(imageGray_, imageGray);

    cv::Mat homography;
    if(state == TRACKING){
        //homography = current_homography;
        homography = TrackingUseOpticalFlow(imageGray);

        if(!FastRefinePoseUsingNCCPatchMatch(imageGray, homography,
                                 last_maker_points, last_image_points)){
            state = NOT_INIT;
            mutex_matched_results.unlock();
            return false;
        }
        mutex_matched_results.unlock();

    } else if(state == NOT_INIT){
        if(!InitPosePatchBrute(imageGray, homography, output)){
            return false;
        }

        last_gray = imageGray;
        state = TRACKING;
    }

    if(show_viewer){
        std::vector<cv::Point2f> imageCorners2D;
        cv::perspectiveTransform(markerInfo->markerCorners2D, imageCorners2D, homography);
        cv::line(output, imageCorners2D[0], imageCorners2D[1], cv::Scalar( 255, 0, 0), 2 );
        cv::line(output, imageCorners2D[1], imageCorners2D[2], cv::Scalar( 255, 0, 0), 2 );
        cv::line(output, imageCorners2D[2], imageCorners2D[3], cv::Scalar( 255, 0, 0), 2 );
        cv::line(output, imageCorners2D[3], imageCorners2D[0], cv::Scalar( 255, 0, 0), 2 );
        for(size_t i = 0 ; i < last_image_points.size(); i ++){
            cv::circle(output, last_image_points[i], 5, cv::Scalar(0, 0, 255), 2);
        }
    }

    current_homography = homography;
    return true;
}

bool TrackerTracker::inBorder(const cv::Point2f &pt)
{
    return BORDER_SIZE <= pt.x && pt.x < WIDTH - BORDER_SIZE
    && BORDER_SIZE <= pt.y && pt.y < HEIGHT - BORDER_SIZE;
}

cv::Mat TrackerTracker::TrackingUseOpticalFlow(cv::Mat &gray)
{
    // points for this frame
    // vMatchedMarkerPoints, vMatchedMapPoints, vMatchedImagePoints
    std::vector<cv::Point2f> new_optical_pts;
    std::vector<cv::Point2f> new_marker_pts;

    std::vector<uchar> status;
    std::vector<float> err;
    cv::calcOpticalFlowPyrLK(last_gray, gray, last_image_points, new_optical_pts, status, err, cv::Size(31, 31), 3);

    int j = 0;
    std::vector<cv::Point2f> last_optical_pts;
    {
        for (unsigned int i = 0; i < status.size(); i++){
            if (status[i] && inBorder(new_optical_pts[i])){
                new_optical_pts[j++] = new_optical_pts[i];
                new_marker_pts.push_back(last_maker_points[i]);
            }
        }
        new_optical_pts.resize(j);
    }

    if(j < LOST_PTS_NUM){
        state = NOT_INIT;
        return current_homography;
    }

    std::vector<uchar> inlierMask;
    cv::Mat homography = cv::findHomography(new_marker_pts, new_optical_pts, CV_RANSAC, 3, inlierMask);

    j = 0;
    for (unsigned int i = 0; i < inlierMask.size(); i++){
        if (inlierMask[i]){
            new_marker_pts[j] = new_marker_pts[i];
            new_optical_pts[j++] = new_optical_pts[i];
        }
    }
    new_marker_pts.resize(j);
    new_optical_pts.resize(j);

    last_gray = gray;
    last_image_points = new_optical_pts;
    last_maker_points = new_marker_pts;

    if(j < LOST_PTS_NUM){
        state = NOT_INIT;
    }

    return homography;
}


bool TrackerTracker::FastRefinePoseUsingNCCPatchMatch(cv::Mat &gray, cv::Mat &homography,
                                            std::vector<cv::Point2f> &matchedMarkerPoints,
                                            std::vector<cv::Point2f> &matchedImagePoints)
{
    matchedMarkerPoints.clear();
    matchedImagePoints.clear();

    // 1. find all the marker points and project them to the image plane
    //random_shuffle(markerPoints.begin(), markerPoints.end());
    unsigned int nPoints = vKeyPoints.size();

    // project marker points to the image
    std::vector<cv::Point2f> projectedPoints;
    cv::perspectiveTransform(vKeyPoints, projectedPoints, homography);

    // 2. search for all the points
    int searchBorder = (fast_match_patch_size - 1) / 2;
    int searchPatch = 2 * fast_match_search_size + 1;

    for (unsigned int i = 0; i < nPoints; i+=fast_interval) {
        cv::Point2f searchCenter = projectedPoints[i];
        if (searchCenter.x < searchBorder || searchCenter.y < searchBorder
                || searchCenter.x >= gray.cols - searchBorder
                || searchCenter.y >= gray.rows - searchBorder) {
            continue;
        }
        // calculate a warp patch for each feature point
        cv::Mat markerPatch = ImageUtils::warpPatch(homography,
                markerInfo->markerImage, vKeyPoints[i], ncc_match_patch_width,
                ncc_match_patch_width);
        cv::Rect searchROI = cv::Rect(searchCenter.x - fast_match_search_size,
                searchCenter.y - fast_match_search_size, searchPatch,
                searchPatch);

        // with perspectiveTransform a feature point is located in a 21*21 area
        // calculate if the warpPath is corresponding with the perspective transformed point
        // use NCC threshold
        ImageUtils::CCoeffPatchFinder patchFinder (markerPatch);
        cv::Point2f matchLoc = patchFinder.findPatch(gray, searchROI, fast_ncc_match_threshold);
        if (matchLoc.x >= 0 && matchLoc.y >= 0) {
            matchedMarkerPoints.push_back(vKeyPoints[i]);
            matchedImagePoints.push_back(matchLoc);
        }
    }

    if (matchedMarkerPoints.size() < min_feature_threshold){
        return false;
    }

    std::vector<uchar> inlierMask;
    homography = cv::findHomography(matchedMarkerPoints, matchedImagePoints, CV_RANSAC, 3, inlierMask);

    int j = 0;
    for (unsigned int i = 0; i < inlierMask.size(); i++){
        if (inlierMask[i]){
            matchedImagePoints[j] = matchedImagePoints[i];
            matchedMarkerPoints[j++] = matchedMarkerPoints[i];
        }
    }
    matchedMarkerPoints.resize(j);
    matchedImagePoints.resize(j);

    return true;
}

bool TrackerTracker::InitPosePatchBrute(cv::Mat &gray, cv::Mat &homography, cv::Mat &output)
{
    if(pBruteFinder->ProcessGray(gray, homography, output)){
        mutex_matched_results.lock();
        if(!RefinePoseUsingNCCPatchMatch(gray, homography, 0.70,
                                 last_maker_points, vMatchedMapPoints, last_image_points)){
            mutex_matched_results.unlock();
            return false;
        }
        mutex_matched_results.unlock();
        return true;
    } else {
        return false;
    }
}



} //namespace
