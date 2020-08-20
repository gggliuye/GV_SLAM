#include "MarkerFinder.h"

namespace BASTIAN
{

void CvMatToQuaternion(cv::Mat R, double Q[])
{
    double trace = R.at<float>(0,0) + R.at<float>(1,1) + R.at<float>(2,2);

    if (trace > 0.0)
    {
        double s = sqrt(trace + 1.0);
        Q[3] = (s * 0.5);
        s = 0.5 / s;
        Q[0] = ((R.at<float>(2,1) - R.at<float>(1,2)) * s);
        Q[1] = ((R.at<float>(0,2) - R.at<float>(2,0)) * s);
        Q[2] = ((R.at<float>(1,0) - R.at<float>(0,1)) * s);
    }
    else
    {
        int i = R.at<float>(0,0) < R.at<float>(1,1) ? (R.at<float>(1,1) < R.at<float>(2,2) ? 2 : 1) : (R.at<float>(0,0) < R.at<float>(2,2) ? 2 : 0);
        int j = (i + 1) % 3;
        int k = (i + 2) % 3;

        double s = sqrt(R.at<float>(i, i) - R.at<float>(j,j) - R.at<float>(k,k) + 1.0);
        Q[i] = s * 0.5;
        s = 0.5 / s;

        Q[3] = (R.at<float>(k,j) - R.at<float>(j,k)) * s;
        Q[j] = (R.at<float>(j,i) + R.at<float>(i,j)) * s;
        Q[k] = (R.at<float>(k,i) + R.at<float>(i,k)) * s;
    }
}

MarkerFinder::MarkerFinder(std::string markerfile, bool view)
{
    if(useFeatureInit){
        mMatcher  = new cv::FlannBasedMatcher(new cv::flann::LshIndexParams(2, 20, 1));
        matcher_bf = new cv::BFMatcher(cv::NORM_HAMMING2, true);
        mORBFeatureExtractor = new ORBextractor(500,1.2,4,20,7);
    }

    loadMarker(markerfile);
    state = NOT_INIT;

    show_viewer = view;
}


MarkerFinder::~MarkerFinder()
{

}

void MarkerFinder::SetCameraParameters(double fx_, double fy_, double cx_, double cy_)
{
    fx = fx_; fy = fy_; cx = cx_; cy = cy_;

    mIntrinsic = cv::Mat::zeros(3, 3, CV_32FC1);
    mIntrinsic.at<float>(0,0) = fx;
    mIntrinsic.at<float>(0,1) = 0;
    mIntrinsic.at<float>(0,2) = cx;
    mIntrinsic.at<float>(1,0) = 0;
    mIntrinsic.at<float>(1,1) = fy;
    mIntrinsic.at<float>(1,2) = cy;
    mIntrinsic.at<float>(2,0) = 0;
    mIntrinsic.at<float>(2,1) = 0;
    mIntrinsic.at<float>(2,2) = 1;

    mDistortionCoeff = cv::Mat::zeros(5,1,CV_32FC1);
}

bool MarkerFinder::InitSLAM(cv::Mat &gray)
{
    cv::Mat homography;
    if(!InitializePoseEstimationWithMarkerInCenter(gray, homography)){
        return false;
    }

    mutex_matched_results.lock();
    if(!RefinePoseUsingNCCPatchMatch(gray, homography, ncc_match_threshold_1,
                             vMatchedMarkerPoints, vMatchedMapPoints, vMatchedImagePoints)){
        mutex_matched_results.unlock();
        return false;
    }

    if(!RefinePoseUsingNCCPatchMatch(gray, homography, ncc_match_threshold_2,
                             vMatchedMarkerPoints, vMatchedMapPoints, vMatchedImagePoints)){
        mutex_matched_results.unlock();
        return false;
    }
    mutex_matched_results.unlock();
    return true;
}

bool MarkerFinder::InitSLAM(cv::Mat &gray, std::vector<cv::Point2f> &matchedMarkerPoints,
                                 std::vector<cv::Point3f> &matchedMapPoints,
                                 std::vector<cv::Point2f> &matchedImagePoints, double *pose)
{
    cv::Mat homography;
    if(!InitializePoseEstimationWithMarkerInCenter(gray, homography)){
        return false;
    }

    if(!RefinePoseUsingNCCPatchMatch(gray, homography, ncc_match_threshold_1,
                             matchedMarkerPoints, matchedMapPoints, matchedImagePoints)){
        mutex_matched_results.unlock();
        return false;
    }

    if(!RefinePoseUsingNCCPatchMatch(gray, homography, ncc_match_threshold_2,
                             matchedMarkerPoints, matchedMapPoints, matchedImagePoints)){
        mutex_matched_results.unlock();
        return false;
    }

    CalculatePose(homography, pose);

    //CalculatePose(matchedMapPoints, matchedImagePoints, pose);

    return true;
}

void MarkerFinder::CalculatePose(cv::Mat homography, double *pose)
{
    std::vector<cv::Point2f> imageCorners2D;
    cv::perspectiveTransform(markerInfo->markerCorners2D, imageCorners2D, homography);

    cv::Mat pnpRotationEuler;
    cv::Mat pnpTranslation;

    if(!cv::solvePnP(markerInfo->markerCorners3D, imageCorners2D,
                mIntrinsic, mDistortionCoeff, pnpRotationEuler, pnpTranslation)){
        std::cout << "PnP failed ! \n";
        return;
    }

    cv::Mat rotMat;
    cv::Rodrigues(pnpRotationEuler, rotMat);

	//std::cout << pnpTranslation << std::endl;

    double* quaternion = new double[4];
    CvMatToQuaternion(rotMat, quaternion);

    pose[0] = pnpTranslation.at<float>(0);
    pose[1] = pnpTranslation.at<float>(1);
    pose[2] = pnpTranslation.at<float>(2);
    pose[3] = quaternion[0];
    pose[4] = quaternion[1];
    pose[5] = quaternion[2];
    pose[6] = quaternion[3];
            //for(int i = 0; i < 7; i++)
            //    std::cout << pose[i] << " ";
            //std::cout << std::endl;
}

void MarkerFinder::CalculatePose(std::vector<cv::Point3f> &matchedMapPoints,
                                 std::vector<cv::Point2f> &matchedImagePoints,  double *pose)
{
    cv::Mat pnpRotationEuler;
    cv::Mat pnpTranslation;

    //std::cerr << matchedMapPoints.size() << " " << matchedImagePoints.size() << "\n";

    if(matchedMapPoints.size() < 4 )
        return;
    if(!cv::solvePnP(matchedMapPoints, matchedImagePoints,
                mIntrinsic, mDistortionCoeff, pnpRotationEuler, pnpTranslation)){
        std::cout << "PnP failed ! \n";
        return;
    }

    cv::Mat rotMat;
    cv::Rodrigues(pnpRotationEuler, rotMat);

	//std::cout << pnpTranslation << std::endl;

    double* quaternion = new double[4];
    CvMatToQuaternion(rotMat, quaternion);

    pose[0] = pnpTranslation.at<float>(0);
    pose[1] = pnpTranslation.at<float>(1);
    pose[2] = pnpTranslation.at<float>(2);
    pose[3] = quaternion[0];
    pose[4] = quaternion[1];
    pose[5] = quaternion[2];
    pose[6] = quaternion[3];
            //for(int i = 0; i < 7; i++)
            //    std::cout << pose[i] << " ";
            //std::cout << std::endl;
}

bool MarkerFinder::ProcessFrame(cv::Mat &image, cv::Mat &output)
{
    output = image.clone();
    cv::Mat imageGray;
    if (image.channels() == 3) {
        cv::cvtColor(image, imageGray, CV_RGB2GRAY);
    } else if (image.channels() == 1){
        imageGray = image;
    }

    cv::Mat homography;
    if(state == TRACKING){
        homography = current_homography;
    }

    if(state == NOT_INIT){
        if(!InitializePoseEstimationWithMarkerInCenter(imageGray, homography)){
            return false;
        }

        if(show_viewer){
            std::vector<cv::Point2f> imageCorners2D;
            cv::perspectiveTransform(markerInfo->markerCorners2D, imageCorners2D, homography);
            cv::line(output, imageCorners2D[0], imageCorners2D[1], cv::Scalar( 0, 255, 0), 2 );
            cv::line(output, imageCorners2D[1], imageCorners2D[2], cv::Scalar( 0, 255, 0), 2 );
            cv::line(output, imageCorners2D[2], imageCorners2D[3], cv::Scalar( 0, 255, 0), 2 );
            cv::line(output, imageCorners2D[3], imageCorners2D[0], cv::Scalar( 0, 255, 0), 2 );
        }
        state = TRACKING;
    }

    mutex_matched_results.lock();
    if(!RefinePoseUsingNCCPatchMatch(imageGray, homography, ncc_match_threshold_1,
                             vMatchedMarkerPoints, vMatchedMapPoints, vMatchedImagePoints)){
        state = NOT_INIT;
        mutex_matched_results.unlock();
        return false;
    }

    if(!RefinePoseUsingNCCPatchMatch(imageGray, homography, ncc_match_threshold_2,
                             vMatchedMarkerPoints, vMatchedMapPoints, vMatchedImagePoints)){
        state = NOT_INIT;
        mutex_matched_results.unlock();
        return false;
    }
    mutex_matched_results.unlock();

    if(show_viewer){
        std::vector<cv::Point2f> imageCorners2D;
        cv::perspectiveTransform(markerInfo->markerCorners2D, imageCorners2D, homography);
        cv::line(output, imageCorners2D[0], imageCorners2D[1], cv::Scalar( 255, 0, 0), 2 );
        cv::line(output, imageCorners2D[1], imageCorners2D[2], cv::Scalar( 255, 0, 0), 2 );
        cv::line(output, imageCorners2D[2], imageCorners2D[3], cv::Scalar( 255, 0, 0), 2 );
        cv::line(output, imageCorners2D[3], imageCorners2D[0], cv::Scalar( 255, 0, 0), 2 );
        for(size_t i = 0 ; i < vMatchedImagePoints.size(); i ++){
            cv::circle(output, vMatchedImagePoints[i], 5, cv::Scalar(0, 0, 255), 2);
        }
    }

    current_homography = homography;
    return true;
}

bool MarkerFinder::GetGuidePoints(std::vector<cv::Point2f> &imageCorners2D)
{
    if(bHaveInited){
        imageCorners2D.clear();
        cv::perspectiveTransform(markerInfo->markerCorners2D, imageCorners2D, init_homography);
        return true;
    } else {
        return false;
    }
}

void MarkerFinder::GetMatchedImagePointsAndMapPoints(std::vector<cv::Point3f> &matchedMapPoints,
                                       std::vector<cv::Point2f> &matchedImagePoints)
{
    mutex_matched_results.lock();
    matchedMapPoints.clear();
    matchedImagePoints.clear();
    matchedMapPoints = vMatchedMapPoints;
    matchedImagePoints = vMatchedImagePoints;
    mutex_matched_results.unlock();
}

void MarkerFinder::GetMarkerInitSize(double &cols, double &rows)
{
    mutex_maker_init_size.lock();
    cols = marker_cols;
    rows = marker_rows;
    mutex_maker_init_size.unlock();
}

bool MarkerFinder::InitializePoseEstimationUsingOrbFeatures(cv::Mat &gray, cv::Mat &homography)
{
    // extract orb features
    std::vector<cv::KeyPoint> keyPoints;
    cv::Mat descriptors;
    mORBFeatureExtractor->GetFeature(gray,keyPoints,descriptors);

    if(keyPoints.size() < min_feature_threshold){
        return false;
    }

    // use knn match to find init matches of points
    std::vector<cv::DMatch> goodMatches;

    std::vector<std::vector<cv::DMatch>> nnMatches;
    mMatcher->knnMatch(descriptors, nnMatches, 1);

    if (nnMatches.size() < min_feature_threshold/2){
        return false;
    }

    // find good matches within the knn match result
    //float pyramidThresholdArray[5] = {95, 95, 95, 96, 98};
    float pyramidThresholdArray[5] = {85, 85, 85, 86, 88};
    for (unsigned int i = 0; i < nnMatches.size(); ++i) {
        std::cout << nnMatches[i].size() << " ";
        for (unsigned int j = 0; j < nnMatches[i].size(); ++j) {
            int octave = markerInfo->keyPoints.at(nnMatches[i][j].trainIdx).octave;
            float pyramidThreshold = 90;
            if (octave < 5)
                pyramidThreshold = pyramidThresholdArray[octave];

            if (nnMatches[i][j].distance < pyramidThreshold) {
                goodMatches.push_back(nnMatches[i][j]);
            }
        }
    }
    std::cout << nnMatches.size() << "\n";

    unsigned int num_good_match = goodMatches.size();
    if (num_good_match < min_feature_threshold/2){
        return false;
    }

    // make the vector of the matched points
    std::vector<cv::Point2f> markerPoints(num_good_match);
    std::vector<cv::Point2f> imagePoints(num_good_match);

    for (unsigned int i = 0; i < num_good_match; ++i) {
        markerPoints[i] = markerInfo->keyPoints[goodMatches[i].trainIdx].pt;
        imagePoints[i] = keyPoints[goodMatches[i].queryIdx].pt;
    }

    std::vector<char> affineMask;
    cv::Mat aff = RansacAffine::getAffineTransform(markerPoints, imagePoints,
        3, 5000, affineMask);

    if (aff.empty()){
        return false;
    }

    homography = cv::Mat::eye(3, 3, aff.type());
    for (int i = 0; i < 2; ++i) {
        for (int j = 0; j < 3; ++j) {
            homography.at<double>(i, j) = aff.at<double>(i, j);
        }
    }

/*
    // calculate the homography again with more points
    std::vector<uchar> mask;
    homography = cv::findHomography(markerPoints, imagePoints, CV_RANSAC, 3, mask, 100);
*/
    if (homography.empty()){
        return false;
    }

    return true;
}


bool MarkerFinder::InitializePoseEstimationUsingOrbFeaturesBF(cv::Mat &gray, cv::Mat &homography)
{
    // extract orb features
    std::vector<cv::KeyPoint> keyPoints;
    cv::Mat descriptors;
    mORBFeatureExtractor->GetFeature(gray,keyPoints,descriptors);

    if(keyPoints.size() < min_feature_threshold){
        return false;
    }

    // use knn match to find init matches of points
    std::vector<cv::DMatch> matches;

    matcher_bf->match(markerInfo->descriptors, descriptors, matches);

    // make the vector of the matched points
    std::vector<cv::Point2f> markerPoints;
    std::vector<cv::Point2f> imagePoints;

    // find good matches within the knn match result
    float pyramidThreshold = 30;
    for (unsigned int i = 0; i < matches.size(); ++i) {
        if (matches[i].distance < pyramidThreshold) {
            //std::cout << matches[i].distance << " ";
            markerPoints.push_back(markerInfo->keyPoints[matches[i].trainIdx].pt);
            imagePoints.push_back(keyPoints[matches[i].queryIdx].pt);
        }
    }

    unsigned int num_good_match = markerPoints.size();
    //std::cout << num_good_match << "\n";
    if (num_good_match < min_feature_threshold/2){
        return false;
    }

    std::vector<char> affineMask;
    cv::Mat aff = RansacAffine::getAffineTransform(markerPoints, imagePoints,
        3, 5000, affineMask);

    if (aff.empty()){
        return false;
    }

    homography = cv::Mat::eye(3, 3, aff.type());
    for (int i = 0; i < 2; ++i) {
        for (int j = 0; j < 3; ++j) {
            homography.at<double>(i, j) = aff.at<double>(i, j);
        }
    }

    return true;
}


bool MarkerFinder::InitializePoseEstimationWithMarkerInCenter(cv::Mat &gray, cv::Mat &homography)
{
    if(bHaveInited){
        homography = init_homography;
        return true;
    }

    mutex_maker_init_size.lock();
    marker_cols = markerInfo->markerImage.cols;
    marker_rows = markerInfo->markerImage.rows;

    double resize_ratio = 0.8;
    while(marker_cols > gray.cols || marker_rows > gray.rows){
        marker_cols = marker_cols * resize_ratio;
        marker_rows = marker_rows * resize_ratio;
    }
    mutex_maker_init_size.unlock();

    // make the vector of the matched points
    std::vector<cv::Point2f> imagePoints(4);

    // top left
    float left = (gray.cols - marker_cols)/2.0;
    float top = (gray.rows - marker_rows)/2.0;
    imagePoints[0] = cv::Point2f(left, top);
    imagePoints[1] = cv::Point2f(left + marker_cols, top);
    imagePoints[2] = cv::Point2f(left + marker_cols, top + marker_rows);
    imagePoints[3] = cv::Point2f(left, top + marker_rows);

    homography = cv::findHomography(markerInfo->markerCorners2D, imagePoints);

    init_homography = homography;
    bHaveInited = true;

    return true;
}

bool MarkerFinder::RefinePoseUsingNCCPatchMatch(cv::Mat &gray, cv::Mat &homography, double ncc_match_threshold,
                                            std::vector<cv::Point2f> &matchedMarkerPoints,
                                            std::vector<cv::Point3f> &matchedMapPoints,
                                            std::vector<cv::Point2f> &matchedImagePoints)
{
    matchedMarkerPoints.clear();
    matchedImagePoints.clear();
    matchedMapPoints.clear();

    // 1. find all the marker points and project them to the image plane
    //random_shuffle(markerPoints.begin(), markerPoints.end());
    unsigned int nPoints = vKeyPoints.size();

    // project marker points to the image
    std::vector<cv::Point2f> projectedPoints;
    cv::perspectiveTransform(vKeyPoints, projectedPoints, homography);

    // 2. search for all the points
    int searchBorder = (ncc_match_patch_width - 1) / 2;
    int searchPatch = 2 * ncc_search_radius + 1;

#ifdef WE_WANT_OPENMP
    #pragma omp parallel for
#endif
    for (unsigned int i = 0; i < nPoints; ++i) {
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
        cv::Rect searchROI = cv::Rect(searchCenter.x - ncc_search_radius,
                searchCenter.y - ncc_search_radius, searchPatch,
                searchPatch);

        // with perspectiveTransform a feature point is located in a 21*21 area
        // calculate if the warpPath is corresponding with the perspective transformed point
        // use NCC threshold
        ImageUtils::CCoeffPatchFinder patchFinder (markerPatch);
        cv::Point2f matchLoc = patchFinder.findPatch(gray, searchROI, ncc_match_threshold);
#ifdef WE_WANT_OPENMP
        #pragma omp critical
#endif
        if (matchLoc.x >= 0 && matchLoc.y >= 0) {
            matchedMarkerPoints.push_back(vKeyPoints[i]);
            matchedImagePoints.push_back(matchLoc);
            matchedMapPoints.push_back(vMapPoints[i]);
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
            matchedMarkerPoints[j] = matchedMarkerPoints[i];
            matchedMapPoints[j++] = matchedMapPoints[i];
        }
    }
    matchedMapPoints.resize(j);
    matchedMarkerPoints.resize(j);
    matchedImagePoints.resize(j);


    return true;
}


void MarkerFinder::loadMarker(std::string markerfile)
{
    cv::Mat des;
    cv::Mat markerImg;
    std::vector<cv::KeyPoint> kps;
    std::vector<cv::Point2f> goodFeatures;
    std::vector<cv::Point2f> marker2d;
    std::vector<cv::Point3f> marker3d;

    std::ifstream inFile(markerfile,std::ios::in|std::ios::binary);

    readFromBinary(inFile,markerImg);
    readFromBinary(inFile,des);
    readFromBinary(inFile,kps);
    read2DPointFromFile(inFile,goodFeatures);
    read2DPointFromFile(inFile,marker2d);
    read3DPointFromFile(inFile, marker3d);

    markerInfo = new ImageMarkerInfo;
    markerInfo->markerImage = markerImg;
    markerInfo->markerCorners2D = marker2d;
    markerInfo->markerCorners3D = marker3d;
    markerInfo->goodFeatures = goodFeatures;
    markerInfo->keyPoints = kps;
    markerInfo->descriptors = des;

    // make map point poses
    float real_width = 2.0 * marker3d[1].x;
    float real_height = 2.0 * marker3d[2].y;
    float pixel_width = markerImg.cols;
    float pixel_height = markerImg.rows;

    if(use_orb_to_refine){
        vMapPoints.resize(kps.size());
        vKeyPoints.resize(kps.size());
        for(size_t i = 0; i < kps.size(); i++){
            float x = float(kps[i].pt.x) * real_width / pixel_width - real_width/2;
            float y = float(kps[i].pt.y) * real_height / pixel_height - real_height/2;
            vMapPoints[i] = cv::Point3f(x, y, 0);
            vKeyPoints[i] = cv::Point2f(kps[i].pt.x, kps[i].pt.y);
        }
    } else {
        vMapPoints.resize(goodFeatures.size());
        vKeyPoints.resize(goodFeatures.size());
        for(size_t i = 0; i < goodFeatures.size(); i++){
            float x = float(goodFeatures[i].x) * real_width / pixel_width - real_width/2;
            float y = float(goodFeatures[i].y) * real_height / pixel_height - real_height/2;
            vMapPoints[i] = cv::Point3f(x, y, 0);
            vKeyPoints[i] = cv::Point2f(goodFeatures[i].x, goodFeatures[i].y);
        }

    }

    std::cout << " [MARKER] loadMarker done ! " << std::endl;
    std::cout << " [MARKER] loadMarker image " << markerImg.cols << "*" << markerImg.rows << std::endl;

    if(useFeatureInit){
        std::vector<cv::Mat> markerDescriptors;
        markerDescriptors.push_back(markerInfo->descriptors);
        mMatcher->clear();
        mMatcher->add(markerDescriptors);
        mMatcher->train();
    }
}


void MarkerFinder::readFromBinary(std::ifstream &inputFile, cv::Mat &des)
{
    int w = 0, h = 0, type = 0;
    int tmp[3];
    inputFile.read((char *)tmp, sizeof(int)*3);
    w = tmp[0]; h = tmp[1]; type = tmp[2];

    cv::Mat m(h,w,type);
    inputFile.read((char *)(m.ptr()),(m.elemSize() * m.total()));
    des = m;
}

void MarkerFinder::readFromBinary(std::ifstream &inputFile, std::vector<cv::KeyPoint> &keyPoint)
{
    int n = 0;
    inputFile.read((char *)&n, sizeof(int));
    cv::KeyPoint kp;
    float tmpPoint[2];
    float tmp[2];
    for (int i = 0; i < n; ++i) {
        inputFile.read((char *)tmpPoint, sizeof(float)*2);
        kp.pt.x = tmpPoint[0];
        kp.pt.y = tmpPoint[1];
        inputFile.read((char *)tmp, sizeof(float)*2);
        kp.size = tmp[0];
        kp.angle = tmp[1];
        inputFile.read((char *)&kp.octave, sizeof(int));
        inputFile.read((char *)&kp.class_id, sizeof(int));
        keyPoint.push_back(kp);
    }
}

void MarkerFinder::read2DPointFromFile(std::ifstream &inputFile, std::vector<cv::Point2f> &p2d)
{
    int n = 0;
    inputFile.read((char *)&n, sizeof(int));

    cv::Point2f p2;
    float tmpPoint[2];
    for (int i = 0; i < n; ++i) {
        inputFile.read((char *)tmpPoint, sizeof(float)*2);
        p2.x = tmpPoint[0];
        p2.y = tmpPoint[1];
        p2d.push_back(p2);
    }
}

void MarkerFinder::read3DPointFromFile(std::ifstream &inputFile, std::vector<cv::Point3f> &p3d)
{
    int n = 0;
    inputFile.read((char *)&n, sizeof(int));

    cv::Point3f p3;
    float tmpPoint[3];
    for (int i = 0; i < n; ++i) {
        inputFile.read((char *)tmpPoint, sizeof(float)*3);
        p3.x = tmpPoint[0];
        p3.y = tmpPoint[1];
        p3.z = tmpPoint[2];
        p3d.push_back(p3);
    }
}

} // namespace BASTIAN
