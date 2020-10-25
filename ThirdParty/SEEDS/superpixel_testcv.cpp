// ****************************************************************************** 
// SEEDS Superpixels
// ******************************************************************************
// Author: Beat Kueng based on Michael Van den Bergh's code
// Contact: vamichae@vision.ee.ethz.ch
//
// This code implements the superpixel method described in:
// M. Van den Bergh, X. Boix, G. Roig, B. de Capitani and L. Van Gool, 
// "SEEDS: Superpixels Extracted via Energy-Driven Sampling",
// ECCV 2012
// 
// Copyright (c) 2012 Michael Van den Bergh (ETH Zurich). All rights reserved.
// ******************************************************************************

#include <vector>
#include <string>

#include "seeds2.h"

#include <cv.h>
#include <highgui.h>
#include <fstream>


#include "helper.h"

#include <ctime>

using namespace std;


std::string getNameFromPathWithoutExtension(std::string path){
  std::string nameWith =  path.substr(path.find_last_of("/\\")+1);
  std::string nameWithout = nameWith.substr(0,nameWith.find_last_of("."));
  return nameWithout;
}


int main(int argc, char* argv[])
{
    char* input_file1;
    if(argc > 1){
        input_file1 = argv[1];   // color image
    } else {
        printf("Error : no filename given as input");
        printf("Usage : %s image_name [number_of_superpixels]\n",argv[0]);
        return -1;
    }

    int NR_SUPERPIXELS = 200;
    if(argc > 2)
        NR_SUPERPIXELS = atoi(argv[2]);

    int numlabels = 10;

    cv::Mat image = cv::imread(input_file1);

    if (image.empty()){
        CV_Error(CV_StsBadArg, "image is empty");
        return -1;
    }
    if (image.type() != CV_8UC3){
        CV_Error(CV_StsBadArg, "image mush have CV_8UC3 type");
        return -1;
    }

    int width = image.cols;
    int height = image.rows;
    int npixels = width * height;
    //  int sz = height*width;
    printf("==> Image loaded. \n");

    std::vector<UINT> ubuff(npixels);
    for (int y = 0; y < height; ++y){
        for (int x = 0; x < width; ++x){
            cv::Vec3b c = image.at<cv::Vec3b>(y, x);

            // image is assumed to have data in BGR order
            UINT b = c[0];
            UINT g = c[1];
            UINT r = c[2];

            ubuff[x + y * width] = b | (g << 8) | (r << 16);
        }
    }



    /*******************************************
    * SEEDS SUPERPIXELS                       *
    *******************************************/
    int NR_BINS = 5; // Number of bins in each histogram channel

    printf("==> Generating SEEDS with %d superpixels\n", NR_SUPERPIXELS);
    SEEDS seeds(width, height, 3, NR_BINS);

    // SEEDS INITIALIZE
    int seed_width = 6; int seed_height = 8; int nr_levels = 4; // parameters to change
    seeds.initialize(seed_width, seed_height, nr_levels);

    // SEEDS PROCESS
    clock_t begin = clock();
    seeds.update_image_ycbcr(&ubuff[0]);
    seeds.iterate();
    clock_t end = clock();

    double elapsed_secs = double(end - begin) / CLOCKS_PER_SEC;
    printf("==> Elapsed time = %lf sec\n", elapsed_secs);
    printf("==> SEEDS produced %d labels\n", seeds.count_superpixels());

    // DRAW SEEDS OUTPUT
    UINT *slabels = seeds.get_labels();
    UINT maxElem = *std::max_element(slabels, slabels + npixels);

    /*******************************************
    * DRAW CONTOURS                           *
    *******************************************/

    std::vector<int> counts(maxElem + 1, 0);
    for (int i = 0; i < npixels; ++i)
        counts[slabels[i]]++;

    std::vector<int> deltas(maxElem + 1);
    int delta = 0;
    for (size_t i = 0; i < counts.size(); ++i)
    {
        if (counts[i] == 0)
            delta++;
        deltas[i] = delta;
    }

    for (int i = 0; i < npixels; ++i)
        slabels[i] -= deltas[slabels[i]];

    cv::Mat labels;
    labels.create(height, width, CV_32SC1);
    for (int y = 0; y < height; ++y)
        for (int x = 0; x < width; ++x)
            labels.at<int>(y, x) = slabels[x + y * width];

    int count_superpixels = maxElem - deltas[maxElem] + 1;
    //printf("==> SEEDS produced %d labels\n", count_superpixels);

    cv::Mat mShow;
    DrawContoursOpencv(labels, mShow, 2);

    std::string imageFileName = getNameFromPathWithoutExtension(std::string(input_file1));
    imageFileName = "./" + imageFileName + "_labels.png";
    printf("==> Saving image %s\n",imageFileName.c_str());
    cv::imwrite(imageFileName, mShow);

    printf(" -- Done! --\n");

    return 0;
}
