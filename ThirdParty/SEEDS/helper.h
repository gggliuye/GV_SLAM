// Helper functions taken from SLIC superpixels source code
// http://ivrg.epfl.ch/supplementary_material/RK_SLICSuperpixels/index.html

#include <opencv2/opencv.hpp>

void DrawContoursOpencv(cv::InputArray _labels, cv::OutputArray _contour, bool thick_line)
{
    cv::Mat labels = _labels.getMat();
    cv::Mat &contour = _contour.getMatRef();

    if (labels.empty())
        CV_Error(CV_StsBadArg, "image is empty");
    if (labels.type() != CV_32SC1)
        CV_Error(CV_StsBadArg, "labels mush have CV_32SC1 type");

    int width = labels.cols;
    int height = labels.rows;

    contour.create(height, width, CV_8UC1);
    contour.setTo(cv::Scalar(0));

    const int dx8[8] = { -1, -1, 0, 1, 1, 1, 0, -1 };
    const int dy8[8] = { 0, -1, -1, -1, 0, 1, 1, 1 };

    for (int j = 0; j < height; j++)
    {
        for (int k = 0; k < width; k++)
        {
            int neighbors = 0;
            for (int i = 0; i < 8; i++)
            {
                int x = k + dx8[i];
                int y = j + dy8[i];

                if( (x >= 0 && x < width) && (y >= 0 && y < height) )
                {
                    if( labels.at<int>(j, k) != labels.at<int>(y, x) )
                    {
                        if( thick_line || !contour.at<uchar>(y, x) )
                            neighbors++;
                    }
                }
            }
            if( neighbors > 1 )
			{
                contour.at<uchar>(j, k) = (uchar)-1;
			}
        }
    }
}

void DrawContoursAroundSegments(UINT* img, UINT* labels, const int& width, const int& height, const UINT& color, bool internal)
{
  const int dx8[8] = {-1, -1,  0,  1, 1, 1, 0, -1};
  const int dy8[8] = { 0, -1, -1, -1, 0, 1, 1,  1};

  int sz = width*height;

  vector<bool> istaken(sz, false);

  int mainindex(0);
  int cind(0);
  for( int j = 0; j < height; j++ )
    {
      for( int k = 0; k < width; k++ )
        {
          int np(0);
          for( int i = 0; i < 8; i++ )
            {
              int x = k + dx8[i];
              int y = j + dy8[i];

              if( (x >= 0 && x < width) && (y >= 0 && y < height) )
                {
                  int index = y*width + x;

				if (internal)
				{
                    {
                      if( labels[mainindex] != labels[index] ) np++;
                    }
				} else {
				  if( false == istaken[index] )//comment this to obtain internal contours
                    {
                      if( labels[mainindex] != labels[index] ) np++;
                    }
				}
                }
            }
          if( np > 1 )
            {
              istaken[mainindex] = true;
              img[mainindex] = color;
              cind++;
            }
          mainindex++;
        }
    }
}

#ifndef WINDOWS


void SaveImage(
               UINT*	ubuff,				// RGB buffer
               const int&			width,				// size
               const int&			height,
               const string&		fileName)			// filename to be given; even if whole path is given, it is still the filename that is used
{
  IplImage* img = cvCreateImage(cvSize(width,height),IPL_DEPTH_8U,3); 
  uchar* pValue;
  int idx = 0;

  for(int j=0;j<img->height;j++)
    for(int i=0;i<img->width;i++)
      {
        pValue = &((uchar*)(img->imageData + img->widthStep*(j)))[(i)*img->nChannels];
        pValue[0] = ubuff[idx] & 0xff;
        pValue[1] = (ubuff[idx] >> 8) & 0xff;
        pValue[2] = (ubuff[idx] >>16) & 0xff;
        idx++;
      }

  //cv::imwrite(fileName, img);
  cvSaveImage(fileName.c_str(),img);
}

void SaveImage_bw(
               UINT*	ubuff,				// RGB buffer
               const int&			width,				// size
               const int&			height,
               const string&		fileName)			// filename to be given; even if whole path is given, it is still the filename that is used
{
  IplImage* img = cvCreateImage(cvSize(width,height),IPL_DEPTH_8U,3); 
  uchar* pValue;
  int idx = 0;

  for(int j=0;j<img->height;j++)
    for(int i=0;i<img->width;i++)
      {
        pValue = &((uchar*)(img->imageData + img->widthStep*(j)))[(i)*img->nChannels];
        pValue[0] = ubuff[idx] & 0xff;
        pValue[1] = ubuff[idx] & 0xff;
        pValue[2] = ubuff[idx] & 0xff;
        idx++;
      }

  //cv::imwrite(fileName, img);
  cvSaveImage(fileName.c_str(),img);
}



#endif
