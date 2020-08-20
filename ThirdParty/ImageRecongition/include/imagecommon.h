#ifndef IMAGECOMMON_H
#define IMAGECOMMON_H

#include <opencv2/core.hpp>


const float inlier_threshold = 3.0f; // Distance threshold to identify inliers
const float nn_match_ratio = 0.9f;   // Nearest neighbor matching ratio

#define EFFECTIVE_NUM 20 //matching effective num, more than 20
#define MARKER_MIN_NUMBER_DESCRIPTORS 100
#define SCALE 0.5
#define MARKER_STANDARD_HEIGHT 480
#define MARKER_STANDARD_WIDTH 640
#define A4_WIDTH 210
#define A4_HEIGHT 297
#define NUM_OF_KEYPOINTS_PER_PYRAMID 300

#define IS_REFINE_FUNDA 1
#define IS_REFINE_MATCHES 1

#define MATCHING_SCORE 0.2
#define MATCH_RATIO 3 //matching used ratio*min_dis
#define VALID_DISTANCE 10.0

#define WIGHT_BUFFER_WIDTH 32
#define WIGHT_BUFFER_HEIGHT 24

#define PATCH_SQUARE_WIDTH 9
#define SEARCH_SQUARE_WIDTH 17
#define PATCH_NCC_THRESHOLD 0.80
#define	PATCH_TRACKER_GOOD_POINT_THRESHOLD 30
#define NUM_OF_MARKER_POINTS_ON_EACH_SCALE 300
#define IMAGE_MARKER_ID_OFFSET 10000
#define POSE_ANGLE_THRESHOLD 70

#define MAX_MARKERS 6

enum ImgMatchRet{
    IMG_ERROR = -3,
    OTHER_ERROR = -2,
    MATCH_BAD = -1,
    OPT_NORMAL = 0
};

enum ImageRating{
    IMG_RATING_1 = 0,
    IMG_RATING_2,
    IMG_RATING_3,
    IMG_RATING_4,
    IMG_RATING_5,
    IMG_RATING
};

struct Dat{
    cv::Mat markerImage;
    std::vector<cv::KeyPoint> kp;
    cv::Mat des;
    std::vector<cv::Point2f> goodFeatures;
    std::vector<cv::Point3f> point_3d;
};

struct MatchingRet{
    float ratio;
    std::string id;
    float pose[4][4];
};

struct ImageMarkerInfo{
    cv::Mat markerImage;
    std::vector<cv::Point2f> markerCorners2D;
    std::vector<cv::Point3f> markerCorners3D;
    std::vector<cv::Point2f> goodFeatures;
    std::vector<cv::KeyPoint> keyPoints;
    cv::Mat descriptors;

    int nFeatures;
};

struct SearchResult {
    int id;
    std::vector<cv::DMatch> matches;
};

struct DetectedMarker{
    int id;
    float pose[16];
//    float pose[7]; //0-2:translation,3-6:Rotation,quaternion
};

#endif // IMAGECOMMON_H
