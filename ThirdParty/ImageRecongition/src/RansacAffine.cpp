#include "RansacAffine.h"

cv::Mat RansacAffine::getAffineTransform(const std::vector<cv::Point2f> &src,
	const std::vector<cv::Point2f> &dst,
	float reprojectionThreshold,
	int maxIteration,
	std::vector<char> &mask) {

	using namespace std;
	using namespace cv;

	int count = static_cast<int>(src.size());

	if (count < 3) {
		return Mat();
	}

	float confidence = 0.995;
	float thresholdSq = reprojectionThreshold * reprojectionThreshold;

	int subset[3];
    Mat M(2, 3, CV_64F), X(6, 1, CV_64F, M.data);
    double a[6 * 6], b[6];
    Mat A(6, 6, CV_64F, a), B(6, 1, CV_64F, b);

	vector<int> indices(src.size());
	for (size_t i = 0; i < src.size(); ++i) {
		indices[i] = i;
	}

	vector<Point2f> proj(src.size());
	vector<char> tmpMask(src.size());
    Mat bestAffine(2, 3, CV_64F);

	int maxGoodCount = 0;
	int iter = 0;
	int i, j, k, goodCount;

	for (iter = 0; iter < maxIteration; iter++) {
		goodCount = 0;
		getRandomSubset(indices, subset);
		for (i = 0; i < 3; i++)
		{
			j = i * 12;
			k = i * 12 + 6;
			a[j] = a[k + 3] = src[subset[i]].x;
			a[j + 1] = a[k + 4] = src[subset[i]].y;
			a[j + 2] = a[k + 5] = 1;
			a[j + 3] = a[j + 4] = a[j + 5] = 0;
			a[k] = a[k + 1] = a[k + 2] = 0;
			b[i * 2] = dst[subset[i]].x;
			b[i * 2 + 1] = dst[subset[i]].y;
		}

        solve(A, B, X);

		goodCount = findInliers(reinterpret_cast<double*>(X.data), src, dst, reprojectionThreshold, tmpMask);

		if (goodCount > maxGoodCount) {
			M.copyTo(bestAffine);
			maxGoodCount = goodCount;
			mask = tmpMask;
			maxIteration = cvRANSACUpdateNumIters(confidence,
				(double)(count - goodCount) / count, 3, maxIteration);
		}
	}

	if (maxGoodCount > 0) {
		return bestAffine;
	}

	return Mat();
}

bool RansacAffine::getRandomSubset(std::vector<int> &indices,
	int subset[3]) {
	//cv::Point2f randSrc[3];
	int n = static_cast<int>(indices.size());
	int i, r;
	for (i = 0; i < 3 && n > 0;) {
		r = rand() % n;
		subset[i] = indices[r];
		//randSrc[i] = src[subset[i]];
		std::swap(indices[r], indices[n - 1]);
		n--;
		i++;
		//if (i == 2) {
		//	bool isColinear = colinearTest(randSrc);
		//	if (!isColinear)
		//		return true;
		//} else {
		//	i++;
		//}
	}
	return true;
}

bool RansacAffine::colinearTest(const cv::Point2f points[]) {
	float d = points[0].x*(points[1].y - points[2].y) + points[1].x * (points[2].y - points[0].y) +
		points[2].x * (points[0].y - points[1].y);
	return  d > -1e-3 && d < 1e-3;
}

int RansacAffine::findInliers(const double *affine,
	const std::vector<cv::Point2f> &src,
	const std::vector<cv::Point2f> &dst,
	float reprojectionThreshold,
	std::vector<char> &mask) {

	int nInliers = 0;
	float dx, dy, err;
	mask.resize(src.size());
	for (size_t i = 0; i < src.size(); ++i) {
		dx = affine[0] * src[i].x + affine[1] * src[i].y + affine[2] - dst[i].x;
		dy = affine[3] * src[i].x + affine[4] * src[i].y + affine[5] - dst[i].y;
		err = dx * dx + dy * dy;
		if (err <= reprojectionThreshold) {
			nInliers++;
			mask[i] = 1;
		} else {
			mask[i] = 0;
		}
	}
	return nInliers;
}
