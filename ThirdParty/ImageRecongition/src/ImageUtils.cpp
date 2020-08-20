#include "ImageUtils.h"
//#include <opencv2/xfeatures2d.hpp>
#include <utility>

#if defined(ANDROID) //|| defined(IPHONE)
//#include <arm_neon.h>
#endif

//cv::Ptr<cv::xfeatures2d::FREAK> ImageUtils::freakExtractor;

std::string type2str(int type) {
  std::string r;

  uchar depth = type & CV_MAT_DEPTH_MASK;
  uchar chans = 1 + (type >> CV_CN_SHIFT);

  switch ( depth ) {
    case CV_8U:  r = "8U"; break;
    case CV_8S:  r = "8S"; break;
    case CV_16U: r = "16U"; break;
    case CV_16S: r = "16S"; break;
    case CV_32S: r = "32S"; break;
    case CV_32F: r = "32F"; break;
    case CV_64F: r = "64F"; break;
    default:     r = "User"; break;
  }

  r += "C";
  r += (chans+'0');

  return r;
}

cv::Mat ImageUtils::warpPatch(cv::Mat homography, cv::Mat src,
        cv::Point2f center, int patchWidth, int patchHeight) {
	cv::Mat tempCenter = cv::Mat::ones(3, 1, homography.type());
	tempCenter.at<double>(0) = center.x;
	tempCenter.at<double>(1) = center.y;
	cv::Mat tempProjectedCenter = homography * tempCenter;
    cv::Point2f projectedCenter(
			tempProjectedCenter.at<double>(0)
					/ tempProjectedCenter.at<double>(2),
			tempProjectedCenter.at<double>(1)
					/ tempProjectedCenter.at<double>(2));

	cv::Mat offsetHomography = cv::Mat::eye(3, 3, homography.type());
    cv::Point2f patchCenter((patchWidth - 1) / 2.0, (patchHeight - 1) / 2.0);
	offsetHomography.at<double>(0, 2) = patchCenter.x - projectedCenter.x;
	offsetHomography.at<double>(1, 2) = patchCenter.y - projectedCenter.y;
	cv::Mat newHomography = offsetHomography * homography;
	newHomography *= 1.0 / newHomography.at<double>(2, 2);

	cv::Mat markerPatch;
	//cv::setNumThreads(1); // set number of threads for parallel_for_
	cv::warpPerspective(src, markerPatch, newHomography,
			cv::Size(patchWidth, patchHeight));

	return markerPatch;
}

bool ImageUtils::isInRange(const cv::Point2f &pt, int width, int height,
		int margin) {
	return pt.x >= margin && pt.y >= margin && pt.x < width - margin
			&& pt.y < height - margin;
}

ImageUtils::CCoeffPatchFinder::CCoeffPatchFinder() :
patchRadius(0), patchSize(0), patchSum(0), patchSqSum(0) {
}

void ImageUtils::CCoeffPatchFinder::makeTemplate(const cv::Mat &inPatch) {
	inPatch.copyTo(patch);
	patchRadius = (patch.rows - 1) / 2;
	patchSize = patch.rows * patch.cols;

	patchSum = 0;
	patchSqSum = 0;

	for (int i = 0; i < patchSize; ++i) {
		patchSum += patch.data[i];
		patchSqSum += patch.data[i] * patch.data[i];
	}

	patchMean = patchSum / patchSize;
}

ImageUtils::CCoeffPatchFinder::CCoeffPatchFinder(cv::Mat inPatch) {
	makeTemplate(inPatch);
}

float ImageUtils::quadraticFitting(float prev, float extreme, float next, bool isFindMin) {
    float a = (next + prev - 2 * extreme);
    float b = (next - prev);
    if ((isFindMin && a <= 0) || (!isFindMin && a >= 0)) {
        return 0;
    }
    return -0.25 * b / a;
}

cv::Point2f ImageUtils::CCoeffPatchFinder::findPatch(cv::Mat &image, cv::Rect roi,
        float threshold) {
    cv::Point2f matchPt(-1, -1);
    float maxMatchScore = -1.0f;
    cv::Mat ccoeffMap(roi.size(), CV_32F, cv::Scalar(0));
    //int patchRadius = (patch.rows - 1) / 2;
    cv::Point searchPt;
	float ccoeff;
    for (searchPt.y = roi.y; searchPt.y < roi.y + roi.height; ++searchPt.y) {
        for (searchPt.x = roi.x; searchPt.x < roi.x + roi.width; ++searchPt.x) {
            //if (!isInRange(searchPt, image.cols, image.rows, patchRadius))
            //    continue;
            ccoeff = getCCoeffAtPoint(image, searchPt);
            ccoeffMap.at<float>(searchPt - cv::Point(roi.x, roi.y)) = ccoeff;
            if (ccoeff > maxMatchScore) {
                matchPt = searchPt;
                maxMatchScore = ccoeff;
            }
        }
    }

    if (maxMatchScore < threshold) {
        matchPt.x = matchPt.y = -1;
    } else {
        int mapX = matchPt.x - roi.x, mapY = matchPt.y - roi.y;
        if (mapX > 1 && mapX < ccoeffMap.cols - 1) {
            matchPt.x += quadraticFitting(ccoeffMap.at<float>(mapY, mapX - 1), ccoeffMap.at<float>(mapY, mapX),
                                          ccoeffMap.at<float>(mapY, mapX + 1), false);
        }
        if (mapY > 1 && mapY < ccoeffMap.rows - 1) {
            matchPt.y += quadraticFitting(ccoeffMap.at<float>(mapY - 1, mapX), ccoeffMap.at<float>(mapY, mapX),
                                          ccoeffMap.at<float>(mapY + 1, mapX), false);
        }
    }
    return matchPt;
}

// bad performance using openmp

float ImageUtils::CCoeffPatchFinder::getCCoeffAtPoint(cv::Mat image,
		cv::Point pt) {
	int imageSum = 0;
	int imageSqSum = 0;
	int imageCrossSum = 0;
	{
		uchar *patchRow, *imageRow;
		uchar d;
		for (int i = 0; i < patch.rows; ++i){
			patchRow = patch.data + i * patch.step;
			imageRow = image.data + (i + pt.y - patchRadius) * image.step
					+ pt.x - patchRadius;
			for (int j = 0; j < patch.cols; ++j){
				d = imageRow[j];
				imageSum += d;
				imageSqSum += d * d;
				imageCrossSum += d * patchRow[j];
			}
		}
	}

    float imageMean = static_cast<float>(imageSum) / patchSize;

    float a = imageCrossSum - imageSum * patchMean;
    float b = (patchSqSum - patchSize * patchMean * patchMean)
			* (imageSqSum - patchSize * imageMean * imageMean);
	return a / sqrt(b);
}

void ImageUtils::halfSample(cv::Mat srcMat, cv::Mat dstMat) {
    cv::resize(srcMat, dstMat, dstMat.size());
}

bool ImageUtils::compareKeypointResponse(const cv::KeyPoint &first, const cv::KeyPoint &second) {
    if (first.response > second.response) {
        return true;
    } else {
        return false;
    }
}


float ImageUtils::findShiTomasiScoreAtPoint(cv::Mat image, int nHalfBoxSize, cv::Point pt) {
	if (!isInRange(pt, image.cols, image.rows, 1))
		return 0.0;

	double dXX = 0;
	double dYY = 0;
	double dXY = 0;
	double dx, dy;
	const uchar* rowData;
	cv::Point2f irStart = pt
		- cv::Point(nHalfBoxSize, nHalfBoxSize);
	cv::Point irEnd = pt + cv::Point(nHalfBoxSize, nHalfBoxSize);

	cv::Point ir;
	int rowSkip = image.step;
	for (ir.y = irStart.y; ir.y <= irEnd.y; ir.y++) {
		rowData = image.ptr(ir.y);
		for (ir.x = irStart.x; ir.x <= irEnd.x; ir.x++) {
			dx = rowData[ir.x + 1] - rowData[ir.x-1];
			dy = rowData[ir.x + rowSkip] - rowData[ir.x - rowSkip];

			dXX += dx * dx;
			dYY += dy * dy;
			dXY += dx * dy;
		}
	}

	int nPixels = (irEnd.x - irStart.x + 1) * (irEnd.y - irStart.y + 1);
	dXX = dXX / (2.0 * nPixels);
	dYY = dYY / (2.0 * nPixels);
	dXY = dXY / (2.0 * nPixels);

	// Find and return smaller eigenvalue:
	return 0.5 * (dXX + dYY - sqrt((dXX + dYY) * (dXX + dYY)
		- 4 * (dXX * dYY - dXY * dXY)));
}

void ImageUtils::blur(const cv::Mat &input, cv::Mat &output) {
	static cv::Mat kernelX, kernelY;
	static bool isInitialized = false;
	if (!isInitialized) {
		kernelX.create(1, 3, CV_32F);
		kernelY.create(3, 1, CV_32F);
		kernelX.at<float>(0) = 1.0f / 3.0f;
		kernelX.at<float>(1) = 1.0f / 3.0f;
		kernelX.at<float>(2) = 1.0f / 3.0f;
		kernelY.at<float>(0) = 1.0f / 3.0f;
		kernelY.at<float>(1) = 1.0f / 3.0f;
		kernelY.at<float>(2) = 1.0f / 3.0f;
		isInitialized = true;
	}
	cv::sepFilter2D(input, output, -1, kernelX, kernelY);
}

void ImageUtils::drawPolygon(cv::Mat &m, const std::vector<cv::Point2f> &polygon, cv::Scalar color, int lineWidth) {
	int n = (int)polygon.size();
	for (int i = 0; i < n; ++i) {
		cv::line(m, polygon[i], polygon[(i + 1) % n], color, lineWidth, CV_AA);
	}
}


std::vector<int> ImageUtils::fastScore(const std::vector<cv::KeyPoint> &corners, int barrier,
	cv::Mat image) {
	if (corners.empty())
		return std::vector<int>();

	static const cv::Point PixelRing[16] = {
		cv::Point(0, 3),cv::Point(1, 3),cv::Point(2, 2),
		cv::Point(3, 1),cv::Point(3, 0),cv::Point(3, -1),
		cv::Point(2, -2),cv::Point(1, -3),cv::Point(0, -3),
	    cv::Point(-1, -3),cv::Point(-2, -2),cv::Point(-3, -1),
		cv::Point(-3, 0),cv::Point(-3, 1),cv::Point(-2, 2),
		cv::Point(-1, 3)
	};

	int pointerDir[16];
	for (int i = 0; i < 16; i++) {
		pointerDir[i] = PixelRing[i].x + PixelRing[i].y * image.cols;
	}
	std::vector<int> scores(corners.size());
	for (unsigned i = 0; i < corners.size(); i++) {
		const cv::Point &pt = corners[i].pt;
		const uchar *imp = image.ptr(pt.y) + pt.x;
		int cb = *imp + barrier;
		int c_b = *imp - barrier;
		int sp = 0, sn = 0;
		for (int j = 0; j < 16; ++j) {
			int p = imp[pointerDir[j]];
			if (p > cb) sp += p - cb;
			else if (p < c_b) sp += c_b - p;
		}
		scores[i] = sp > sn ? sp : sn;
	}
	return scores;
}

void ImageUtils::nonMaxSuppression(const std::vector<cv::KeyPoint> &corners,
	const std::vector<int> &scores, std::vector<cv::KeyPoint> &nonMaxCorners) {
	nonMaxCorners.clear();
	if (corners.empty())
		return;
	// Find where each row begins
	// (the corners are output in raster scan order). A beginning of -1 signifies
	// that there are no corners on that row.
	int last_row = corners.back().pt.y;
	std::vector<int> row_start(last_row + 1, -1);

	int prev_row = -1;
	for (unsigned int i = 0; i < corners.size(); i++) {
		if (corners[i].pt.y != prev_row) {
			row_start[corners[i].pt.y] = i;
			prev_row = corners[i].pt.y;
		}
	}
	//Point above points (roughly) to the pixel above the one of interest, if there
	//is a feature there.
	int point_above = 0;
	int point_below = 0;
	const int sz = (int)corners.size();
	for (int iCorner = 0; iCorner < sz; iCorner++){
		int score = scores[iCorner];
		cv::Point2f pos = corners[iCorner].pt;
		//Check left
		if (iCorner > 0)
			if (corners[iCorner - 1].pt == pos - cv::Point2f(1, 0) && scores[iCorner - 1] > score)
				continue;
		//Check right
		if (iCorner < (sz - 1))
			if (corners[iCorner + 1].pt == pos + cv::Point2f(1, 0) && scores[iCorner + 1] > score)
				continue;
		//Check above (if there is a valid row above)
		if (pos.y != 0 && row_start[pos.y - 1] != -1){
			//Make sure that current point_above is one
			//row above.
			if (corners[point_above].pt.y < pos.y - 1)
				point_above = row_start[pos.y - 1];

			//Make point_above point to the first of the pixels above the current point,
			//if it exists.
			for (; corners[point_above].pt.y < pos.y && corners[point_above].pt.x < pos.x - 1; point_above++){
			}

			for (int i = point_above; corners[i].pt.y < pos.y && corners[i].pt.x <= pos.x + 1; i++){
				int x = corners[i].pt.x;
				if ((x == pos.x - 1 || x == pos.x || x == pos.x + 1) && scores[i] > score)
					goto cont;
			}
		}
		//Check below (if there is anything below)
		if (pos.y != last_row && row_start[pos.y + 1] != -1 && point_below < sz) {
			if (corners[point_below].pt.y < pos.y + 1)
				point_below = row_start[pos.y + 1];
			// Make point below point to one of the pixels belowthe current point, if it
			// exists.
			for (; point_below < sz && corners[point_below].pt.y == pos.y + 1 && corners[point_below].pt.x < pos.x - 1;
				point_below++){
			}
			for (int i = point_below; i < sz && corners[i].pt.y == pos.y + 1 && corners[i].pt.x <= pos.x + 1; i++){
				int x = corners[i].pt.x;
				if ((x == pos.x - 1 || x == pos.x || x == pos.x + 1) && scores[i] > score)
					goto cont;
			}
		}
		nonMaxCorners.push_back(corners[iCorner]);
	cont:
		;
	}
}

void ImageUtils::drawText(cv::Mat &image, const std::string & text, const cv::Point &origin, const cv::Scalar &color) {
	int nLine = 0;
	int lineHeight = 15;
	std::stringstream textStream(text);
	char line[1024];

	while (!textStream.eof()) {
		textStream.getline(line, 1024);
		cv::Point textOrigin(10, 10 + lineHeight * (nLine + 1));
		cv::putText(image, line, textOrigin, 1, 1, cv::Scalar(255, 0, 0));
		nLine++;
	}
}
