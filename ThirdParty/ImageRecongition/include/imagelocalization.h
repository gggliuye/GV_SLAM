#ifndef ImageLocalization_H
#define ImageLocalization_H

#include <map>
#include <string>
#include <chrono>
#include <pthread.h>
#include <opencv2/core.hpp>
#include <opencv2/features2d.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
//#include <opencv2/xfeatures2d.hpp>
#include "imagecommon.h"
#include "orbextractor.h"

class ImageLocalization
{

public:
    ImageLocalization();
    ~ImageLocalization();

    /*
    *  init the intrinsic camera parameters
    */
    int Init(float fx = 493.0167F, float fy = 491.55953F, float cx = 317.97856F, float cy = 242.392F);

    //for ios use
    //bool ImgMatchingToXml(cv::Mat img, /*std::string file2,*/ std::vector<DetectedMarker> &detecMarker);

    //Android use
    bool ImgMatchingToXml(cv::Mat &img, std::string &detecMarker,float* pose);


    static bool compareSearchResult(const SearchResult &r1, const SearchResult &r2) {
        return r1.matches.size() > r2.matches.size();
    }

    cv::Mat imageToDraw;

private:

    void convertColor(cv::Mat src, cv::Mat &dst);

    void initCornerArrays(cv::Mat markerImage, float real_width, std::vector<cv::Point2f> &marker2dPoint, std::vector<cv::Point3f> &marker3dPoint);

    void initORBKeypointAndDescriptors(cv::Mat frame, std::vector<cv::KeyPoint>& keypoints, cv::Mat &descriptors);

    //void initKeypointAndDescriptors(cv::Mat frame, std::vector<cv::KeyPoint>& keypoints, cv::Mat &descriptors, int nLevels = 7);

    void getORBKeypointsAndDescriptors(cv::Mat image,  std::vector<cv::KeyPoint> &keyPoints, cv::Mat &descriptors);
 
    /*
    void getKeypointsAndDescriptors(int nLevels,cv::Mat image,
                                    std::vector<cv::KeyPoint> &keypoints,
                                    cv::Mat &descriptors,
                                    int maxKeyPointsPerLevel = 200);
    */

    bool search(const cv::Mat frame, std::vector<cv::KeyPoint> &imageKeypoints);

    bool verifyMatches(cv::Mat frame, std::vector<cv::KeyPoint> frameKeyPoints, const std::vector<SearchResult> & foundMarkers,cv::Range range);

    bool verify(cv::Mat homography, cv::Mat frame, const ImageMarkerInfo &markerInfo);

    bool verifyByPatches(cv::Mat homography, cv::Mat frame, const ImageMarkerInfo &markerInfo,
                         std::vector<cv::Point2f> &matchedMarkerPoints,std::vector<cv::Point2f> &matchedImagePoints);

    cv::Mat estimateHomographyFromAffine(cv::Mat frame, ImageMarkerInfo markerInfo, std::vector<cv::KeyPoint> frameKeyPoints, const std::vector<cv::DMatch> &matches);

    bool CalculatePose(const std::vector<cv::Point2f>& corners2D,
                    const std::vector<cv::Point3f>& corners3D,
                    const cv::Mat intrinsic,
                    const cv::Mat distortionCoeff);

    void resizeMarker(cv::Mat &src, cv::Mat &dst);

    void resizeImg(cv::Mat &src, cv::Mat &dst);

private:
    // camera parameters
    cv::Mat mIntrinsic; 
    cv::Mat mDistortionCoeff; 

    double imageResizeRatio;
    double markerResizeRatio;

    cv::DescriptorMatcher *mMatcher;
    std::map<std::string, Dat> mapDat;

    bool isDescriptorsChanged;

    cv::Mat refinedHomography;

    std::vector<cv::Point2f> imageCorners2D;

    cv::Mat poseRotation;
    cv::Mat poseTranslation;
    cv::Mat pnpRotationEuler;
    cv::Mat pnpTranslation;

    std::map<int, ImageMarkerInfo> markerInfoMap;
    int markerIdCounter;
    std::vector<int> loadedIds;
    std::vector<cv::Mat> loadedDescriptors;

    std::vector<SearchResult> imageResult;
    std::vector<DetectedMarker> detectedMarkers;

    pthread_mutex_t detectedVectorMutex;

private:
    // feature extractors
    //cv::Ptr<cv::xfeatures2d::FREAK> freakExtractor;

    ORBextractor * mORBFeatureExtractor;

	///////////////////////////////////////
	///////// verify multi-marker  ////////
	///////////////////////////////////////


public: 

    static void * VerifyMatchesByThread(void *args);

    struct VerifyMatchArgs {
        ImageLocalization *detector;
        const cv::Mat frame;
        const std::vector<cv::KeyPoint> &imageKeyPoints;
        const std::vector<SearchResult> & foundMarkers;
        cv::Range range;

        VerifyMatchArgs(ImageLocalization *_detector,
            const cv::Mat &_frame,
            const std::vector<cv::KeyPoint> &_imageKeyPoints,
            const std::vector<SearchResult> & _foundMarkers,
            cv::Range &_range):
            detector(_detector),frame(_frame),
            imageKeyPoints(_imageKeyPoints), foundMarkers(_foundMarkers),
            range(_range){

        }
    };

	///////////////////////////////////////
	///////// gestion with marker  ////////
	///////////////////////////////////////

public:
    /*
     * input:  input image file path
     * input:  input a desired file path to output
     * return : 0 -> success, -1 -> failed
     */
    int saveImgMarkerInfo(std::string imgfile, float real_width, std::string savepath);
    int loadMarker(std::string markerfile);

    void unloadAllMarkers();
    void unloadSingleMarker(int id);

private:
    //write Keypoint to binary file
    void writeToBinary(const std::vector<cv::KeyPoint> &kp, std::ofstream &outputFile);

    //write descriptors to binary file
    void writeToBinary(const cv::Mat &des, std::ofstream &outputFile);

    //write 2d point to binary file
    void write2DPointToBinary(const std::vector<cv::Point2f> p2d, std::ofstream &outputFile);

    //write A4 3d points to binary file
    void write3DPointToBinary(const std::vector<cv::Point3f> p3d, std::ofstream &outputFile);

    //read descriptors
    void readFromBinary(std::ifstream &inputFile, cv::Mat &des);

    //read keypoint
    void readFromBinary(std::ifstream &inputFile, std::vector<cv::KeyPoint> &keyPoint);

    //////  read point2f  /////
    //read A4 3d points
    void read2DPointFromFile(std::ifstream &inputFile, std::vector<cv::Point2f> &p2d);
    //read A4 3d points
    void read3DPointFromFile(std::ifstream &inputFile, std::vector<cv::Point3f> &p3d);


};

class SortIndices {
    const std::vector<int> &scores;
    public:
        SortIndices(const std::vector<int> &_scores) :
            scores(_scores) {}
        bool operator() (int i, int j) const {
            return scores[i] > scores[j];
        }
};

class TicToc{
private:
    std::chrono::steady_clock::time_point t1;

public: 
    TicToc(){
        t1 = std::chrono::steady_clock::now();
    }

    ~TicToc(){}

    void reset(){
        t1 = std::chrono::steady_clock::now();
    }

    double Count(){
        std::chrono::steady_clock::time_point t2 = std::chrono::steady_clock::now();
        return std::chrono::duration_cast<std::chrono::duration<double> >(t2 - t1).count();
    }


};

#endif // ImageLocalization_H
