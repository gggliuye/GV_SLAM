#include <string>
#include <iostream>
//#include <unistd.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <fstream>
#define _USE_MATH_DEFINES
#include <math.h>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
//#include <opencv2/xfeatures2d.hpp>
//#define ANDROID
#ifdef ANDROID
#include <android/log.h>
#define  LOG_TAG    "imagelocalization"
#define  LOGI(...)  __android_log_print(ANDROID_LOG_INFO, LOG_TAG, __VA_ARGS__)
#define LOGE(...)  __android_log_print(ANDROID_LOG_ERROR, LOG_TAG, __VA_ARGS__)
#define LOGI(...)  __android_log_print(ANDROID_LOG_INFO, LOG_TAG, __VA_ARGS__)
//#include <json.h>
#endif
#include "imagelocalization.h"
#include "RansacAffine.h"
#include "ImageUtils.h"

//#define DEBUG 1

using namespace std;
using namespace cv;

ImageLocalization::ImageLocalization():isDescriptorsChanged(false),markerIdCounter(0)
{
    pthread_mutex_init(&detectedVectorMutex, NULL);
    mMatcher  = new cv::FlannBasedMatcher(new cv::flann::LshIndexParams(2, 20, 1));
    //freakExtractor = cv::xfeatures2d::FREAK::create(true, true, 22.0f, 1);
    mORBFeatureExtractor = new ORBextractor(500,1.2,4,20,7);
}

ImageLocalization::~ImageLocalization()
{
    if (mMatcher){
        delete mMatcher;
        mMatcher = NULL;
    }
    unloadAllMarkers();
    pthread_mutex_destroy(&detectedVectorMutex);
}

int ImageLocalization::Init(float fx, float fy, float cx, float cy)
{
    int ret = 0;
    //init intrinsic parameters
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

    std::cout<<"Init Done!"<<endl;

    return ret;
}


//#ifdef ANDROID
bool ImageLocalization::ImgMatchingToXml(cv::Mat &img, std::string &detecMarker, float * pose)
{
    bool ret = false;
    detectedMarkers.clear();

    //TicToc prepareImageT;
    //-- Initial images
    cv::Mat gray;
    convertColor(img, gray);

    //image resize to 640*480
    //resizeImg(gray,gray);

    //std::cout << " [ANALYSIS] prepare image takes " << prepareImageT.Count() << " s " << std::endl;

    TicToc searchT;
    std::vector<cv::KeyPoint> imageKeypoints;
    ret = search(gray, imageKeypoints);
    if (!ret){
        //std::cout<< " [ERROR] search false"<<std::endl;
        return ret;
    }

    //std::cout << " [ANALYSIS] search image takes " << searchT.Count() << " s " << std::endl;

    searchT.reset();
    if (markerInfoMap.size() < 2){
        ret = verifyMatches(gray,imageKeypoints,imageResult,cv::Range(0,imageResult.size()));
    }
    
    //std::cout << " [ANALYSIS] verify image takes " << searchT.Count() << " s " << std::endl;
    //std::cout << " [SUCCESS] detected marker :  " << detectedMarkers.size() << std::endl << std::endl;

    if (detectedMarkers.size() > 0){
        // for test, only need one marker
        if (pose)
            memcpy(pose, detectedMarkers[0].pose, sizeof(detectedMarkers[0].pose));
    }
    else{
        for(int k = 0; k < 16; k++){
            pose[k] = 0;
        }
    }
    return ret;
}
//#endif

//////////////////////////////////////////////////////////////////
/////////////// load, unload, and make maker /////////////////////
////////////////////////////////////////////////////////////////// 

int ImageLocalization::saveImgMarkerInfo(std::string imgfile, float real_width, std::string savepath)
{
    int ret = OPT_NORMAL;

    cv::Mat image = imread(imgfile.c_str(), IMREAD_GRAYSCALE);
    if(!image.data){
        std::cout << " [ERROR] fail to load image file : " << imgfile << endl;
        return IMG_ERROR;
    }
    cv::Mat gray;
    std::vector<cv::Point2f> marker2d;
    std::vector<cv::Point3f> marker3d;
    marker2d.resize(4);
    marker3d.resize(4);

    convertColor(image,gray);
    resizeMarker(gray,gray);
    initCornerArrays(gray,real_width,marker2d,marker3d);

    //-- Detect keypoint based on ORB
    std::vector<cv::KeyPoint> keypoints;
    cv::Mat descriptors;
    //    m_FeatureExtractor.GetFeature(image,Mat(), keypoints, descriptors);
    //initKeypointAndDescriptors(gray, keypoints, descriptors);
    initORBKeypointAndDescriptors(gray, keypoints, descriptors);

    std::cout<<" [ImageMarkerInfo] keypoints:"<<keypoints.size()<<std::endl;
    std::cout<<" [ImageMarkerInfo] des:"<<descriptors.size()<<std::endl;

    std::vector<cv::Point2f> goodFeatures;
    goodFeaturesToTrack(gray, goodFeatures, 400, 0.015, 10);
    cv::TermCriteria criteria = cv::TermCriteria(cv::TermCriteria::MAX_ITER + cv::TermCriteria::EPS,40,0.01);
    cv::cornerSubPix(gray, goodFeatures, cv::Size(5, 5), cv::Size(-1, -1), criteria);

    if (keypoints.size() > MARKER_MIN_NUMBER_DESCRIPTORS){
        std::ofstream outFile (savepath, std::ios::binary);
        if (!outFile){
            cout<<" [ImageMarkerInfo] cannot open output folder: "<<savepath<<endl;;
        }
        //write image
        writeToBinary(gray,outFile);
        //write descriptors
        writeToBinary(descriptors,outFile);
        //write keypoints
        writeToBinary(keypoints,outFile);
        //write goodfeatures
        write2DPointToBinary(goodFeatures,outFile);
        //write 2d corner points
        write2DPointToBinary(marker2d,outFile);
        //write 3d points
        write3DPointToBinary(marker3d,outFile);
        outFile.close();
        cout<<" [ImageMarkerInfo] write done!"<<endl;
    }
    else{
        cout<<" [ImageMarkerInfo] features too less:"<<keypoints.size()<<endl;
    }

    return ret;
}

int ImageLocalization::loadMarker(std::string markerfile)
{
    std::vector<cv::Point3f> marker3d;
    std::vector<cv::Point2f> marker2d;
    std::vector<cv::Point2f> goodFeatures;
    cv::Mat des;
    std::vector<cv::KeyPoint> kps;
    std::ifstream inFile(markerfile,ios::in|ios::binary);

    cv::Mat markerImg;
    readFromBinary(inFile,markerImg);
    readFromBinary(inFile,des);
    readFromBinary(inFile,kps);
    read2DPointFromFile(inFile,goodFeatures);
    read2DPointFromFile(inFile,marker2d);
    read3DPointFromFile(inFile, marker3d);

    ImageMarkerInfo markerInfo;
    markerInfo.markerImage = markerImg;
    markerInfo.markerCorners2D = marker2d;
    markerInfo.markerCorners3D = marker3d;
    markerInfo.goodFeatures = goodFeatures;
    markerInfo.keyPoints = kps;
    markerInfo.descriptors = des;

    int markerId = markerIdCounter + IMAGE_MARKER_ID_OFFSET;
    std::cout<<" [MARKER] marker ID:"<<markerId<<std::endl;
    markerInfoMap.insert(std::pair<int, ImageMarkerInfo>(markerId, markerInfo));
    markerIdCounter++;
    loadedIds.push_back(markerId);
    loadedDescriptors.push_back(markerInfo.descriptors);

    isDescriptorsChanged = true;

    //std::vector<cv::Point2f> markerPoints = markerInfo.goodFeatures;
    //random_shuffle(markerInfo.goodFeatures.begin(), markerInfo.goodFeatures.end());

    std::cout<<" [MARKER] loadMarker done!"<<std::endl;
    return markerId;
}

void ImageLocalization::unloadAllMarkers()
{
    std::cout<<" [MARKER] unload all markers "<<std::endl;

    markerInfoMap.clear();
    loadedIds.clear();
    loadedDescriptors.clear();
    markerIdCounter = 0;
    isDescriptorsChanged = true;
}

void ImageLocalization::unloadSingleMarker(int id)
{
    if (!markerInfoMap.count(id)) {
        std::cout<<" [Unload marker] [ERROR] ID "<<id<<" doesn't exist in the dictionary"<<std::endl;
        return;
    }

    std::cout<<" [Unload marker] : "<<id<<std::endl;
    markerInfoMap.erase(id);

    for (unsigned int i = 0; i < loadedIds.size(); ++i) {
        if (loadedIds[i] == id) {
            loadedIds.erase(loadedIds.begin() + i);
            loadedDescriptors.erase(loadedDescriptors.begin() + i);
        }
    }

    isDescriptorsChanged = true;
}


void * ImageLocalization::VerifyMatchesByThread(void *args) {
    std::cout<<"thread process"<<std::endl;
    VerifyMatchArgs *vmArgs = reinterpret_cast<VerifyMatchArgs*>(args);

    vmArgs->detector->verifyMatches(vmArgs->frame, vmArgs->imageKeyPoints,vmArgs->foundMarkers,vmArgs->range);
    return NULL;
}

void ImageLocalization::initCornerArrays(cv::Mat markerImage, float real_width, std::vector<cv::Point2f> &marker2dPoint, std::vector<cv::Point3f> &marker3dPoint)
{
    float w = markerImage.cols;
    float h = markerImage.rows;

    float read_height = real_width * h / w;
    std::cout<<"[MarkerInfo] resized marker : " << real_width << " x " << read_height << " ratio: " << markerResizeRatio << std::endl;
/*
    double inverseRatio = 1 / markerResizeRatio;
    w = w * inverseRatio;
    h = h * inverseRatio;
*/
    marker2dPoint.resize(4);
    marker3dPoint.resize(4);

    marker2dPoint[0] = cv::Point2f(0, 0);
    marker2dPoint[1] = cv::Point2f(w, 0);
    marker2dPoint[2] = cv::Point2f(w, h);
    marker2dPoint[3] = cv::Point2f(0, h);

    marker3dPoint[0] = cv::Point3f(-real_width/2, -read_height/2, 0);
    marker3dPoint[1] = cv::Point3f(real_width/2, -read_height/2, 0);
    marker3dPoint[2] = cv::Point3f(real_width/2, read_height/2, 0);
    marker3dPoint[3] = cv::Point3f(-real_width/2, read_height/2, 0);
    for (int i = 0; i < 4; i++){
        std::cout<<"2d:["<<marker2dPoint[i].x<<","<<marker2dPoint[i].y<<"]"<<std::endl;
    }
    for (int i = 0; i < 4; i++){
        std::cout<<"3d:["<<marker3dPoint[i].x<<","<<marker3dPoint[i].y<<","<<marker3dPoint[i].z<<"]"<<std::endl;
    }
}

void ImageLocalization::initORBKeypointAndDescriptors(cv::Mat frame, std::vector<cv::KeyPoint>& keypoints, cv::Mat &descriptors)
{

    mORBFeatureExtractor = new ORBextractor(200,1.2,4,20,7);
    getORBKeypointsAndDescriptors(frame, keypoints, descriptors);

}

void ImageLocalization::getORBKeypointsAndDescriptors(cv::Mat image,
                                                std::vector<cv::KeyPoint> &keyPoints,
                                                cv::Mat &descriptors)
{
    mORBFeatureExtractor->GetFeature(image,keyPoints,descriptors);
}

bool ImageLocalization::search(const cv::Mat frame, std::vector<cv::KeyPoint> &imageKeypoints)
{
    int markerCount = static_cast<int>(loadedIds.size());
    if (markerCount == 0){
        std::cout << " [ERROR] NO marker loaded yet. "<< std::endl;
        return false;
    }

    //TicToc timer;
    imageResult.clear();
    // Detect keypoint
    cv::Mat descriptors_q;
 
    getORBKeypointsAndDescriptors(frame, imageKeypoints, descriptors_q);
    //std::cout<<" [KeyPoints] image Keypoints size is : "<<imageKeypoints.size() << ", time takes " << timer.Count() <<endl;

    //abnormal processing
    if (imageKeypoints.size() < EFFECTIVE_NUM){
        //cout<<" [FAILED] too few points. keypoints size :"<<imageKeypoints.size()<<endl;
        return false;
    }

    //timer.reset();
    std::vector<cv::DMatch> matches;
    std::vector< std::vector<cv::DMatch> > nnMatches;
    if (isDescriptorsChanged){
        mMatcher->clear();
        mMatcher->add(loadedDescriptors);
        mMatcher->train();
        isDescriptorsChanged = false;
    }

    mMatcher->knnMatch(descriptors_q, nnMatches, 1);
    //std::cout<<" [Detector] nnMatches size:"<<nnMatches.size()<< ", time takes " << timer.Count()<<std::endl;

    if (nnMatches.size() < EFFECTIVE_NUM/2){
        //cout<<" [FAILED] matches num:"<<matches.size()<<endl;
        return false;
    }

    imageResult.resize(markerCount);
    for (int i = 0; i < markerCount; ++i){
        imageResult[i].id = loadedIds[i];
        imageResult[i].matches.clear();
    }

    //timer.reset();
    int octave = 0;
    float pyramidThreshold;
    int idx = 0;
    for (unsigned int i = 0; i<nnMatches.size(); ++i) {
        for (unsigned int j = 0; j<nnMatches[i].size(); ++j) {
            octave = markerInfoMap[loadedIds[nnMatches[i][j].imgIdx]].keyPoints.at(nnMatches[i][j].trainIdx).octave;
            pyramidThreshold = 100;
            if (octave <= 2)
                pyramidThreshold = 95;
            else if (octave == 3)
                pyramidThreshold = 96;
            else if (octave == 4)
                pyramidThreshold = 98;
            else if (octave >= 5)
                pyramidThreshold = 100;

            if (nnMatches[i][j].distance < pyramidThreshold) {
                idx = nnMatches[i][j].imgIdx;
                imageResult[idx].matches.push_back(nnMatches[i][j]);
            }
        }
    }
    int maxCandidates = min(markerCount, MAX_MARKERS);
    std::sort(imageResult.begin(), imageResult.end(), compareSearchResult);

    imageResult.resize(maxCandidates);

    std::random_shuffle(imageResult.begin(), imageResult.end());
    //std::cout<<" [MATCH] imageResult: "<<imageResult.size()<< ", time takes " << timer.Count()<<std::endl;
    return true;
}

bool ImageLocalization::verify(Mat homography, Mat frame, const ImageMarkerInfo &markerInfo)
{
    //refinedHomography = homography;

    std::vector<cv::Point2f> matchedMarkerPoints, matchedImagePoints;
    if (!verifyByPatches(homography, frame, markerInfo, matchedMarkerPoints, matchedImagePoints)) {
        //std::cout<<" [VERIFY FAILED] Verifying marker: PT_Search FAILED"<<std::endl;
        return false;
    }
    //std::cout<<" [VERIFY SUCCESS] Verifying marker: PT_Search PASS ]"<<std::endl;

    std::vector<uchar> inlierMask (matchedMarkerPoints.size());
    refinedHomography = cv::findHomography(matchedMarkerPoints,matchedImagePoints, CV_RANSAC, 3, inlierMask);

    cv::perspectiveTransform(markerInfo.markerCorners2D, imageCorners2D,refinedHomography);

    line(imageToDraw, imageCorners2D[0], imageCorners2D[1], Scalar(0, 255, 0), 4 );
    line(imageToDraw, imageCorners2D[1], imageCorners2D[2], Scalar( 0, 255, 0), 4 );
    line(imageToDraw, imageCorners2D[2], imageCorners2D[3], Scalar( 0, 255, 0), 4 );
    line(imageToDraw, imageCorners2D[3], imageCorners2D[0], Scalar( 0, 255, 0), 4 );

    if (!CalculatePose(imageCorners2D, markerInfo.markerCorners3D, mIntrinsic, mDistortionCoeff)){
        //std::cout << " [VERIFY FAILED] PnP solve failed. " << std::endl;
        return false;
    }
    return true;
}


// it is the same process as in affline transform calculation 
// should be write into a same function , which can be called multiple times
// the principle is the same, project the features into the image and find match position by NCC
bool ImageLocalization::verifyByPatches(cv::Mat homography, cv::Mat frame,const ImageMarkerInfo &markerInfo,
           std::vector<cv::Point2f> &matchedMarkerPoints,std::vector<cv::Point2f> &matchedImagePoints)
{
    // TODO: no need to do it for each frame, we can shuffle them before processing

    // 1. find all the marker points and project them to the image plane
    std::vector<cv::Point2f> markerPoints = markerInfo.goodFeatures;
    random_shuffle(markerPoints.begin(), markerPoints.end());

    int nPoints = static_cast<int>(markerPoints.size());
    // project marker points to the image
    std::vector<cv::Point2f> projectedPoints;
    cv::perspectiveTransform(markerPoints, projectedPoints, homography);

    // 2. search for all the points
    int searchRadius = (SEARCH_SQUARE_WIDTH - 1) / 2;

    matchedMarkerPoints.clear();
    matchedImagePoints.clear();

    unsigned int nPointsRequired = nPoints * PATCH_TRACKER_GOOD_POINT_THRESHOLD / 100;

    cv::Point2f searchCenter;
    cv::Mat markerPatch;
    cv::Rect searchROI;
    cv::Point2f matchLoc;
    for (int i = 0; i < nPoints; ++i) {
        searchCenter = projectedPoints[i];
        if (searchCenter.x < searchRadius || searchCenter.y < searchRadius
                || searchCenter.x >= frame.cols - searchRadius
                || searchCenter.y >= frame.rows - searchRadius) {
            continue;
        }
        // calculate a warp patch for each feature point
        markerPatch = ImageUtils::warpPatch(homography,
                markerInfo.markerImage, markerPoints[i], PATCH_SQUARE_WIDTH,
                PATCH_SQUARE_WIDTH);
        searchROI = cv::Rect(searchCenter.x - searchRadius,
                searchCenter.y - searchRadius, SEARCH_SQUARE_WIDTH,
                SEARCH_SQUARE_WIDTH);

        // with perspectiveTransform a feature point is located in a 17*17 area
        // calculate if the warpPath is corresponding with the perspective transformed point
        // use NCC threshold
        ImageUtils::CCoeffPatchFinder patchFinder (markerPatch);
        matchLoc = patchFinder.findPatch(frame, searchROI, PATCH_NCC_THRESHOLD);
        if (matchLoc.x >= 0 && matchLoc.y >= 0) {
            matchedMarkerPoints.push_back(markerPoints[i]);
            matchedImagePoints.push_back(matchLoc);
        }
    }

    if (matchedMarkerPoints.size() > nPointsRequired && matchedImagePoints.size() > 15)
    {
        return true;
    }

    return false;
}


#if 1
cv::Mat ImageLocalization::estimateHomographyFromAffine(cv::Mat frame, ImageMarkerInfo markerInfo, std::vector<cv::KeyPoint> frameKeyPoints, const std::vector<cv::DMatch> &matches)
{
    cv::Mat H;
    if (matches.size() < 4) {
        return H;
    }

    // remake the points for fitting types
    // TODO: reduce this part
    std::vector<cv::Point2f> markerPoints(matches.size());
    std::vector<cv::Point2f> imagePoints(matches.size());

    for (unsigned i = 0; i < matches.size(); ++i) {
        markerPoints[i] = markerInfo.keyPoints[matches[i].trainIdx].pt;
        imagePoints[i] = frameKeyPoints[matches[i].queryIdx].pt;
    }

    // affline estimation of the homography matrix
    // TODO: check the efficience
    std::vector<char> affineMask, maske;
    TicToc test;
    cv::Mat aff = RansacAffine::getAffineTransform(markerPoints, imagePoints,
        3, 5000, affineMask);
    //std::cout << "          ---- HOMOGRAPHY JK : " << test.Count() << std::endl;

    if (aff.empty()){
        return cv::Mat();
    }

    // find the inliners 
    std::vector<cv::Point2f> foundMarkerPoints, foundImagePoints;
    for (unsigned i = 0; i < affineMask.size(); ++i) {
        if (affineMask[i]) {
            foundMarkerPoints.push_back(markerPoints[i]);
            foundImagePoints.push_back(imagePoints[i]);
        }
    }
    if (foundMarkerPoints.size() < 5) {
        return cv::Mat();
    }
    cv::Mat homography = cv::Mat::eye(3, 3, aff.type());
    for (int i = 0; i < 2; ++i) {
        for (int j = 0; j < 3; ++j) {
            homography.at<double>(i, j) = aff.at<double>(i, j);
        }
    }


    // project the feature points to the image frame
    std::vector<cv::Point2f> projectedPoints;
    cv::transform(markerInfo.goodFeatures, projectedPoints, aff);

    // use good features, find match points by project and use NCC patch match
    // takes a bit more time by with much higher percise
    // define the search region
    int searchRadius = (SEARCH_SQUARE_WIDTH - 1) / 2;
    cv::Point2f searchCenter;
    cv::Mat markerPatch;
    cv::Rect searchROI;
    cv::Point2f matchLoc;
    for (size_t i = 0; i < projectedPoints.size(); i++)
    {
	// if the projected point is outside the image -> skip 
        searchCenter = projectedPoints[i];
        if (!ImageUtils::isInRange(searchCenter, frame.cols, frame.rows, searchRadius)) {
            continue;
        }

        
        markerPatch = ImageUtils::warpPatch(homography,
            markerInfo.markerImage, markerInfo.goodFeatures[i], PATCH_SQUARE_WIDTH,
            PATCH_SQUARE_WIDTH);

        searchROI.x = searchCenter.x - searchRadius;
        searchROI.y = searchCenter.y - searchRadius;
        searchROI.height = searchROI.width = SEARCH_SQUARE_WIDTH;

        ImageUtils::CCoeffPatchFinder patchFinder(markerPatch);
        matchLoc = patchFinder.findPatch(frame, searchROI, PATCH_NCC_THRESHOLD);
        if (matchLoc.x >= 0 && matchLoc.y >= 0) {
            foundMarkerPoints.push_back(markerInfo.goodFeatures[i]);
            foundImagePoints.push_back(matchLoc);
        }
    }

    if (foundMarkerPoints.size() < 5) {
        return cv::Mat();
    }

    // calculate the homography again with more points
    std::vector<uchar> mask;
    homography = cv::findHomography(foundMarkerPoints, foundImagePoints, CV_RANSAC, 3, mask, 100);

    return homography;
}

#endif

bool ImageLocalization::CalculatePose(const std::vector<cv::Point2f>& corners2D,
                                const std::vector<cv::Point3f>& corners3D,
                                const cv::Mat intrinsic,
                                const cv::Mat distortionCoeff)
{
    if(!cv::solvePnP(corners3D, corners2D,intrinsic, distortionCoeff, pnpRotationEuler, pnpTranslation))
        return false;
    return true;
}


bool ImageLocalization::verifyMatches(cv::Mat frame, std::vector<cv::KeyPoint> frameKeyPoints, const std::vector<SearchResult> & foundMarkers,cv::Range range)
{
    int id = 0;
    cv::Mat homography;
    for (int i = range.start; i < range.end; i++){
        id = foundMarkers[i].id;

        std::map<int,ImageMarkerInfo>::iterator iter;
        iter = markerInfoMap.find(id);
        if (iter == markerInfoMap.end()){
            //std::cout<<" [VERIFY FAILED] verifyMatches Not find marker id:"<<id<<std::endl;
            return false;
        }

        TicToc timer;
        homography = estimateHomographyFromAffine(frame,iter->second,frameKeyPoints,foundMarkers[i].matches);
        //std::cout << " [verify TIME] estimate homography time : " << timer.Count() << " s" << std::endl;
        if (homography.empty()) {
            //std::cout << " [VERIFY FAILED] call estimateHomographyFromAffine homography empty" << std::endl;
            return false;
        }

        timer.reset();
        bool isGood = verify(homography, frame, iter->second);
        //std::cout << " [verify TIME] verify time : " << timer.Count() << " s" << std::endl;

        if (!isGood) {
            //std::cout<<" [VERIFY FAILED] verified marker "<< id <<" failed"<<std::endl;
            continue;
        }
        else {
            //std::cout<<" [VERIFY SUCCESS] verified marker "<< id <<std::endl;
            DetectedMarker marker;
            marker.id = id;

            double inverseRatio = 1 / imageResizeRatio;
            marker.pose[0] = pnpRotationEuler.at<double>(0);
            marker.pose[1] = pnpRotationEuler.at<double>(1);
            marker.pose[2] = pnpRotationEuler.at<double>(2);
            marker.pose[3] = 0;
            marker.pose[4] = pnpTranslation.at<double>(0) * inverseRatio;
            marker.pose[5] = pnpTranslation.at<double>(1) * inverseRatio;
            marker.pose[6] = pnpTranslation.at<double>(2) * inverseRatio;

            for(int i = 0; i < 7; i ++){
                std::cout << marker.pose[i] << " ";
            }
            //std::cout << " ratio: " << inverseRatio << std::endl;

            pthread_mutex_lock(&detectedVectorMutex);
            detectedMarkers.push_back(marker);
            pthread_mutex_unlock(&detectedVectorMutex);
        }
    }

    return true;
}


////////////////////////////////////////////////////////////////////////
///////////// read(write) from (to) binary file ////////////////////////
////////////////////////////////////////////////////////////////////////

void ImageLocalization::writeToBinary(const std::vector<KeyPoint> &kp, ofstream &outputFile)
{
    int n = kp.size();
    cout<<"write kp:"<<n<<endl;
    outputFile.write((char*)&n, sizeof(int));
    for (int i = 0; i < n; i++){
        outputFile.write((char*)&kp[i].pt.x, sizeof(float));
        outputFile.write((char*)&kp[i].pt.y, sizeof(float));
        outputFile.write((char*)&kp[i].size, sizeof(float));
        outputFile.write((char*)&kp[i].angle, sizeof(float));
        outputFile.write((char*)&kp[i].octave, sizeof(int));
        outputFile.write((char*)&kp[i].class_id, sizeof(int));
    }
}

void ImageLocalization::writeToBinary(const Mat &des, ofstream &outputFile)
{
    int w = des.cols;
    int h = des.rows;
    int type = des.type();
    outputFile.write((char*)&w, sizeof(int));
    outputFile.write((char*)&h, sizeof(int));
    outputFile.write((char*)&type, sizeof(int));
    outputFile.write((char*)des.ptr(), des.total() * des.elemSize());
}

void ImageLocalization::write2DPointToBinary(const std::vector<cv::Point2f> p2d, std::ofstream &outputFile)
{
    int n = p2d.size();
    outputFile.write((char*)&n, sizeof(int));
    for (int i = 0; i < n; i++){
        outputFile.write((char*)&p2d[i].x, sizeof(float));
        outputFile.write((char*)&p2d[i].y, sizeof(float));
    }
}

void ImageLocalization::write3DPointToBinary(const std::vector<cv::Point3f> p3d, std::ofstream &outputFile)
{
    int n = p3d.size();
    outputFile.write((char*)&n, sizeof(int));
    for (int i = 0; i < n; i++){
        outputFile.write((char*)&p3d[i].x, sizeof(float));
        outputFile.write((char*)&p3d[i].y, sizeof(float));
        outputFile.write((char*)&p3d[i].z, sizeof(float));
    }
}

void ImageLocalization::readFromBinary(std::ifstream &inputFile, Mat &des)
{
    int w = 0, h = 0, type = 0;
    int tmp[3];
    inputFile.read((char *)tmp, sizeof(int)*3);
    w = tmp[0]; h = tmp[1]; type = tmp[2];

    cv::Mat m(h,w,type);
    inputFile.read((char *)(m.ptr()),(m.elemSize() * m.total()));
    des = m;
}

void ImageLocalization::readFromBinary(std::ifstream &inputFile, std::vector<KeyPoint> &keyPoint)
{
    int n = 0;
    inputFile.read((char *)&n, sizeof(int));
    KeyPoint kp;
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

void ImageLocalization::read2DPointFromFile(std::ifstream &inputFile, std::vector<cv::Point2f> &p2d)
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

void ImageLocalization::read3DPointFromFile(std::ifstream &inputFile, std::vector<cv::Point3f> &p3d)
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

void ImageLocalization::resizeMarker(cv::Mat &src, cv::Mat &dst)
{
    markerResizeRatio = 1.0;
    cv::Mat tmp = src.clone();
    while(tmp.cols > 640 || tmp.rows > 480){
        cv::resize(tmp, tmp, cv::Size(src.cols*0.8,src.rows*0.8));
        markerResizeRatio *= 0.8;
    }
    dst = tmp;
}

void ImageLocalization::resizeImg(cv::Mat &src, cv::Mat &dst)
{
    imageResizeRatio = std::max(
                MARKER_STANDARD_HEIGHT * 1.0 / std::min(src.cols, src.rows),
                MARKER_STANDARD_WIDTH * 1.0 / std::max(src.cols, src.rows));

    cv::resize(src, dst, Size(), imageResizeRatio, imageResizeRatio);
}

void ImageLocalization::convertColor(cv::Mat src, cv::Mat &dst)

{
    if (src.channels() == 3) {
        cv::cvtColor(src, dst, CV_RGB2GRAY);
    } else if (src.channels() == 1){
        src.copyTo(dst);
    }
    else{
       //TODO covert RGBA image
    }
}
