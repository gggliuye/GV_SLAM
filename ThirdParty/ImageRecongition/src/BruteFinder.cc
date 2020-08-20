#include "BruteFinder.h"



namespace BASTIAN
{

BruteFinder::BruteFinder(cv::Mat &imageMarker_, std::vector<cv::Point2f> vMarkerCorners2D_, bool view)
{
    imageMarker = imageMarker_;
    vMarkerCorners2D = vMarkerCorners2D_;
    bView = view;

    vScales.reserve(nLevel);
    vScalesInv.reserve(nLevel);
    vMarkerPyramid.reserve(nLevel);
    vMarkerPatchSum.reserve(nLevel);
    vMarkerPatchSqSum.reserve(nLevel);
    vMarkerPatchMean.reserve(nLevel);
    vMarkerPatchSize.reserve(nLevel);
    double scale_i = rescale_ratio;
    for(int i = 0; i < nLevel; i++){
        scale_i *= dScale;
        vScales.push_back(scale_i);
        vScalesInv.push_back(1/scale_i);

        int col_desire = COLS * scale_i;
        int row_desire = col_desire * imageMarker.rows / imageMarker.cols;
        cv::Mat scale_marker;
        cv::resize(imageMarker, scale_marker, cv::Size(col_desire, row_desire));
        cv::GaussianBlur(scale_marker, scale_marker, cv::Size(gaussian_kernel, gaussian_kernel),0, 0);
        vMarkerPyramid.push_back(scale_marker);
        //std::cout << " -- scale : " << scale_i << " marker pyramid size : " << col_desire << "*" << row_desire << "\n";

    	int patchSize = scale_marker.rows * scale_marker.cols;
    	float patchSum = 0;
    	float patchSqSum = 0;
    	for (int i = 0; i < patchSize; ++i) {
    		patchSum += scale_marker.data[i];
    		patchSqSum += scale_marker.data[i] * scale_marker.data[i];
    	}
    	float patchMean = patchSum / (float)patchSize;
        vMarkerPatchSize.push_back(patchSize);
        vMarkerPatchSum.push_back(patchSum);
        vMarkerPatchSqSum.push_back(patchSqSum);
        vMarkerPatchMean.push_back(patchMean);
    }

    COLS = COLS * rescale_ratio;
    ROWS = ROWS * rescale_ratio;
}

bool BruteFinder::ProcessGray(cv::Mat &gray, cv::Mat &homography, cv::Mat &show_image)
{
    cv::Mat small_blur_gray;
    cv::resize(gray, small_blur_gray, cv::Size(gray.cols*rescale_ratio, gray.rows*rescale_ratio));
    cv::GaussianBlur(small_blur_gray, small_blur_gray, cv::Size(gaussian_kernel, gaussian_kernel),0, 0);

    // loop for all the image pyramids
    cv::Point2f bestPoint;
    int best_level = -1;
    float best_score = -1;
    for(int i = 0; i < nLevel; i++){
        cv::Point2f point_level;
        float coeff = NccCalculation(small_blur_gray, i, show_image, point_level);
        if( coeff > best_score){
            best_score = coeff;
            bestPoint = point_level;
            best_level = i;
        }
    }
    //std::cout << coeff << "\n";
    if(best_score > NCC_THRESHOLD){
        //calculate homography
        std::vector<cv::Point2f> imageCorners2D;
        imageCorners2D.resize(4);
        int level = best_level;
        imageCorners2D[0] = (bestPoint - cv::Point2f(vMarkerPyramid[level].cols/2, vMarkerPyramid[level].rows/2))/rescale_ratio;
        imageCorners2D[1] = (bestPoint + cv::Point2f(vMarkerPyramid[level].cols/2, -vMarkerPyramid[level].rows/2))/rescale_ratio;
        imageCorners2D[2] = (bestPoint + cv::Point2f(vMarkerPyramid[level].cols/2, vMarkerPyramid[level].rows/2))/rescale_ratio;
        imageCorners2D[3] = (bestPoint + cv::Point2f(-vMarkerPyramid[level].cols/2, vMarkerPyramid[level].rows/2))/rescale_ratio;

        homography = cv::findHomography(vMarkerCorners2D, imageCorners2D);
        if(bView)
            DrawRectangle(show_image, imageCorners2D, best_level, best_score);

        return true;
    }

    return false;
}

float BruteFinder::NccCalculation(cv::Mat &image, int level, cv::Mat &show_image, cv::Point2f &best_point)
{
    float best_ncc = -1;
    int interval = base_interval * vScalesInv[level];
    int cols_res = (image.cols - vMarkerPyramid[level].cols) / interval;
    int rows_res = (image.rows - vMarkerPyramid[level].rows) / interval;
    //cv::Mat ccoeffMap(cv::Size(cols_res, rows_res), CV_32F, cv::Scalar(0));
    for(int col0 = 0; col0 < cols_res; col0++){
        for(int row0 = 0; row0 < rows_res; row0++){
            float ncc = NccOnce(image, vMarkerPyramid[level], col0*interval, row0*interval,
                vMarkerPatchSize[level], vMarkerPatchSum[level], vMarkerPatchSqSum[level], vMarkerPatchMean[level]);
            //std::cout << col0 << " " << row0 << " " << ncc << std::endl;
            //ccoeffMap.at<float>(row0, col0) = ncc;
            if(ncc > best_ncc){
                best_ncc = ncc;
                best_point.x = col0*interval+vMarkerPyramid[level].cols/2;
                best_point.y = row0*interval+vMarkerPyramid[level].rows/2;
            }
        }
    }

    return best_ncc;
}

float BruteFinder::NccOnce(cv::Mat &image, cv::Mat &marker, int col0, int row0,
    int patchSize, float patchSum, float patchSqSum, float patchMean)
{
    int imageSum = 0;
    int imageSqSum = 0;
    int imageCrossSum = 0;
    {
        uchar *patchRow, *imageRow;
        uchar d;
        for (int i = 0; i < marker.rows; ++i){
            patchRow = marker.data + i * marker.step;
            imageRow = image.data + (i + row0) * image.step + col0;
            for (int j = 0; j < marker.cols; ++j){
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

void BruteFinder::DrawRectangle(cv::Mat &output, std::vector<cv::Point2f> &imageCorners2D, int level, float best_score)
{
    cv::line(output, imageCorners2D[0], imageCorners2D[1], cv::Scalar( 0, 255, 0), 2 );
    cv::line(output, imageCorners2D[1], imageCorners2D[2], cv::Scalar( 0, 255, 0), 2 );
    cv::line(output, imageCorners2D[2], imageCorners2D[3], cv::Scalar( 0, 255, 0), 2 );
    cv::line(output, imageCorners2D[3], imageCorners2D[0], cv::Scalar( 0, 255, 0), 2 );

    cv::Point pt_text = cv::Point(imageCorners2D[0].x, imageCorners2D[0].y + 20);
    cv::putText(output, "NCC:"+std::to_string(best_score), pt_text, cv::FONT_HERSHEY_SIMPLEX, 0.8, cv::Scalar(255, 50, 50), 2, CV_AA);
}

} // namespace
