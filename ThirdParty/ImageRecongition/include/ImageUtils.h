#ifndef ARFULLSERIES_UTILS_IMAGEUTILS_H_
#define ARFULLSERIES_UTILS_IMAGEUTILS_H_

#include <vector>
//#include <opencv2/opencv.hpp>
#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
//#include <opencv2/xfeatures2d.hpp>

#ifdef USE_OPENMP
#include <omp.h>
#endif

//#define WE_WANT_OPENMP

std::string type2str(int type);

class ImageUtils {
public:
	static cv::Mat warpPatch(cv::Mat homography, cv::Mat src,
            cv::Point2f center, int patchWidth, int patchHeight);

    static bool isInRange(const cv::Point2f &pt, int width, int height,
			int margin = 0);

    template <typename T>
    static bool pointLineSameSideTest(const cv::Point &pt,
			const T &A, const T &B,
			const T &thisSide);

	template<typename T>
    static bool pointQuadrangleTest(cv::Point pt, const std::vector<T> & corners);

    static void halfSample (cv::Mat srcMat, cv::Mat dstMat);

    template<typename T>
    static T findQuadrangleCenter(const std::vector<T> &corners);

    static float quadraticFitting(float prev, float extreme, float next, bool isFindMin);

    static void drawText(cv::Mat &image, const std::string & text, const cv::Point &origin, const cv::Scalar &color);

    static bool compareKeypointResponse(const cv::KeyPoint &first, const cv::KeyPoint &second);

    static float findShiTomasiScoreAtPoint(cv::Mat image, int nHalfBoxSize, cv::Point pt);

	class CCoeffPatchFinder {
	public:
		CCoeffPatchFinder();
		CCoeffPatchFinder(cv::Mat inPatch);

        cv::Point2f findPatch(cv::Mat &image, cv::Rect roi, float threshold);
        float getCCoeffAtPoint(cv::Mat image, cv::Point pt);


		void makeTemplate(const cv::Mat &inPatch);

	private:
		cv::Mat patch;
		int patchRadius;
		int patchSize;

        float patchSum;
        float patchSqSum;
        float patchMean;
	};

	/* functions migrated from CVD library*/
	static std::vector<int> fastScore(const std::vector<cv::KeyPoint> &corners, int barrier,
		cv::Mat image);
	static void nonMaxSuppression(const std::vector<cv::KeyPoint> &corners,
		const std::vector<int> &scores, std::vector<cv::KeyPoint> &nonMaxCorners);
    static inline cv::Point2f project(const cv::Point3f &pt3) {
        return cv::Point2f(pt3.x / pt3.z, pt3.y / pt3.z);
	}
    static inline cv::Point2f project(const cv::Mat &pt3) {
        return cv::Point2f(pt3.at<float>(0) / pt3.at<float>(2), pt3.at<float>(1) / pt3.at<float>(2));
	}
    static inline cv::Point3f unproject(const cv::Point2f &pt2) {
        return cv::Point3f(pt2.x, pt2.y, 1.0f);
	}
    static inline cv::Point2f matToPoint(const cv::Mat &m) {
        return cv::Point2f(m.at<float>(0), m.at<float>(1));
	}

	static void blur(const cv::Mat &input, cv::Mat &output);

	template<typename T>
	static float bilinearSample(const cv::Mat &m, const cv::Point2f &pt) {
		cv::Mat s;
		cv::getRectSubPix(m, cv::Size(1, 1), pt, s, CV_32F);
		return s.at<float>(0, 0);
	}

	static void drawPolygon(cv::Mat &m, const std::vector<cv::Point2f> &polygon, cv::Scalar color, int lineWidth);

private:
	//static cv::Ptr<cv::xfeatures2d::FREAK> freakExtractor;
};
/*
template <typename T>
bool ImageUtils::pointLineSameSideTest(const cv::Point &pt,
                                       const T &A, const T &B,
                                       const T &thisSide) {
    float z1 = (B.x - A.x) * (pt.y - A.y) - (B.y - A.y) * (pt.x - A.x);
    float z2 = (B.x - A.x) * (thisSide.y - A.y) - (B.y - A.y) * (thisSide.x - A.x);
    return z1 * z2 >= 0;
}

template <typename C>
bool ImageUtils::pointQuadrangleTest(cv::Point pt,
                                     const std::vector<C> & corners) {
    return (pointLineSameSideTest(pt, corners[0], corners[1], corners[2])
            && pointLineSameSideTest(pt, corners[1], corners[2], corners[3])
            && pointLineSameSideTest(pt, corners[2], corners[3], corners[0])
            && pointLineSameSideTest(pt, corners[3], corners[0], corners[1]));
}

template <typename T>
T ImageUtils::findQuadrangleCenter(const std::vector<T> &corners) {
    T center;
    center.x = ((corners[3].x - corners[1].x) * ((corners[2].x - corners[0].x) * corners[0].y - (corners[2].y - corners[0].y)*corners[0].x)
                - (corners[2].x - corners[0].x) * ((corners[3].x - corners[1].x) * corners[1].y - (corners[3].y - corners[1].y)*corners[1].x))
    / ((corners[2].x - corners[0].x)*(corners[3].y - corners[1].y) - (corners[3].x - corners[1].x)*(corners[2].y - corners[0].y));
    center.y = (center.x - corners[0].x) * (corners[2].y - corners[0].y) / (corners[2].x - corners[0].x) + corners[0].y;

    return center;
}
*/
#endif // ARFULLSERIES_UTILS_IMAGEUTILS_H_
