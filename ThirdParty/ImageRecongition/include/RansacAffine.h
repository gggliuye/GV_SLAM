#ifndef ARFULLSERIES_UTILS_RANSACAFFINE_H_
#define ARFULLSERIES_UTILS_RANSACAFFINE_H_

//#include <opencv2/opencv.hpp>
#include <opencv2/calib3d.hpp>
#include <opencv2/core.hpp>
#include <vector>

class RansacAffine {
public:
	static cv::Mat getAffineTransform(const std::vector<cv::Point2f> &src,
		const std::vector<cv::Point2f> &dst,
		float reprojectionThreshold,
		int maxIteration,
		std::vector<char> &mask);

private:
	static bool getRandomSubset(std::vector<int> &indices,
		int subset [3]);
	static inline bool colinearTest(const cv::Point2f points[]);
	static int findInliers(const double *affine,
		const std::vector<cv::Point2f> &src,
		const std::vector<cv::Point2f> &dst,
		float reprojectionThreshold,
		std::vector<char> &mask);
};

#endif // ARFULLSERIES_UTILS_RANSACAFFINE_H_
