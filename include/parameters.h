#ifndef BASTIAN_PARAMETERS_H
#define BASTIAN_PARAMETERS_H

// for feature tracker
const int BORDER_SIZE = 3;
const int MIN_DIST = 40;
const int MIN_DIST_TRACK = 10;
const int MIN_DIST_MARKER = 7;

// fundamental check threshold
const float F_THRESHOLD = 2.0; //(pixel)

// maximum of feature in a frame
const int MAX_CNT = 200;
// if few points tracked, the system is lost
const int MIN_TRACKED_PTS = 30;

// for keyframe selection
const int KEYFRAME_TRACKED_POINTS = 20;
const double KEYFRAME_PARALLAX = 20;


// local optimization frames
const int LOCAL_NUM = 20;
const double RANSAC_LOSS_THRESHOLD = 200;







#endif
