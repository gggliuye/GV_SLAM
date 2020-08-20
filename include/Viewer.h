#ifndef BASTIAN_VIEWER_H
#define BASTIAN_VIEWER_H

#include "Tracking.h"
#include "Map.h"

#include <pangolin/pangolin.h>

namespace BASTIAN
{

class Tracker;

class Viewer
{
public:
    Viewer(Tracker *pTracker_, GMap *pGMap_, PinholeCamera *pPinholeCamera_);
    ~Viewer(){}

    void Run();

    void SetFPS(double fps_);

private:
    void DrawMapPoints();
    void DrawKeyFrames();
    void DrawCube(const float &size,const float x, const float y, const float z);
    void DrawCube(cv::Mat &image, Eigen::Matrix4d &Tcw);
    void DrawFPS(cv::Mat &image);

private:
    Tracker *pTracker;
    GMap *pGMap;
    PinholeCamera *pPinholeCamera;

    std::mutex m_fps;
    double fps = 0.0;
    Eigen::Matrix3d intrinsic;

    int imageWidth = 640;
    int imageHeight = 480;
    double dPointSize = 3;
    float dKeyFrameSize = 0.04;
    float size_cube = 0.1;
    float size_cube_half = 0.05;
    float graph_line_width = 0.9;

    bool draw_graph = true;
};

} // namsapce BASTIAN

#endif // BASTIAN_VIEWER_H
