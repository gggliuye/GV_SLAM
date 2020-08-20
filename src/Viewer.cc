#include "Viewer.h"



namespace BASTIAN
{

Viewer::Viewer(Tracker *pTracker_, GMap *pGMap_, PinholeCamera* pPinholeCamera_):
       pTracker(pTracker_), pGMap(pGMap_), pPinholeCamera(pPinholeCamera_)
{
    intrinsic << pPinholeCamera->fx, 0, pPinholeCamera->cx,
                 0, pPinholeCamera->fy, pPinholeCamera->cy,
                 0, 0, 1;
}

void Viewer::SetFPS(double fps_)
{
    std::unique_lock<std::mutex> lock(m_fps);
    fps = fps_;
}

void Viewer::Run()
{
    pangolin::CreateWindowAndBind("GVSLAM: Map Viewer",1024,768);

    // 3D Mouse handler requires depth testing to be enabled
    glEnable(GL_DEPTH_TEST);

    // Issue specific OpenGl we might need
    glEnable (GL_BLEND);
    glBlendFunc (GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);

    pangolin::CreatePanel("menu").SetBounds(0.0,1.0,0.0,pangolin::Attach::Pix(175));
    pangolin::Var<bool> menuFollowCamera("menu.Follow Camera",true,true);
    pangolin::Var<bool> menuShowPoints("menu.Show Points",true,true);
    pangolin::Var<bool> menuShowKeyFrames("menu.Show KeyFrames",true,true);

    // Define Camera Render Object (for view / scene browsing)
    pangolin::OpenGlRenderState s_cam(
                pangolin::ProjectionMatrix(1024,768,500,500,512,389,0.1,1000),
                pangolin::ModelViewLookAt(0, -0.7, -1.8, 0,0,0,0.0,-1.0, 0.0)
                );

    // Add named OpenGL viewport to window and provide 3D Handler
    pangolin::View& d_cam = pangolin::CreateDisplay()
            .SetBounds(0.0, 1.0, pangolin::Attach::Pix(175), 1.0, -1024.0f/768.0f)
            .SetHandler(new pangolin::Handler3D(s_cam));

    cv::namedWindow("GVSLAM: Current Frame");
    while(1)
    {
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

        d_cam.Activate(s_cam);
        glClearColor(1.0f,1.0f,1.0f,1.0f);

        glColor3f(0, 0, 1);
        pangolin::glDrawAxis(3);

        DrawMapPoints();

        DrawKeyFrames();

        //DrawCube(size_cube,0,0,0);

        pangolin::FinishFrame();

        cv::Mat im;
        if(pTracker->GetImageShow(im)){
            Eigen::Matrix4d Tcw;
            if(pTracker->GetCurrentPose(Tcw)){
                DrawCube(im, Tcw);
                //std::cout << Tcw << std::endl;
            }
            DrawFPS(im);
            cv::imshow("GVSLAM: Current Frame", im);
            cv::waitKey(5);
        }
    }
}

void Viewer::DrawMapPoints()
{
    glPointSize(dPointSize);
    glBegin(GL_POINTS);
    glColor3f(0.0,0.0,0.0);
    std::vector<GMapPoint*> vMappoints = pGMap->GetAllMapPoints();
    for(size_t i = 0; i < vMappoints.size() ; i++){
        if(vMappoints[i]->GetBadFlag())
            continue;
        if(!vMappoints[i]->GetbTriangulated())
            continue;
        Eigen::Vector3d pos = vMappoints[i]->GetPose();
        glVertex3f(pos(0), pos(1), pos(2));
    }
    glEnd();

}

void Viewer::DrawKeyFrames()
{
    const float &w = dKeyFrameSize;
    const float h = dKeyFrameSize * 0.75;
    const float z = dKeyFrameSize * 0.6;

    std::vector<GKeyFrame*> vKeyframes = pGMap->GetAllKeyFrames();

    Eigen::Vector3d last_frame(0,0,0);

    for(size_t i = 0; i<vKeyframes.size(); i++)
    {
        GKeyFrame* pKF = vKeyframes[i];
        cv::Mat Twc = pKF->GetPoseTwc();

        pangolin::OpenGlMatrix currentT;
        for(int i = 0; i < 4; i++){
            for(int j = 0; j < 4; j++){
                currentT.m[4*j + i] = Twc.at<float>(i,j);
            }
        }

        glPushMatrix();

        glMultMatrixd(currentT.m);

        glLineWidth(1);
        glColor3f(0.0f,0.0f,1.0f);
        glBegin(GL_LINES);
        glVertex3f(0,0,0);
        glVertex3f(w,h,z);
        glVertex3f(0,0,0);
        glVertex3f(w,-h,z);
        glVertex3f(0,0,0);
        glVertex3f(-w,-h,z);
        glVertex3f(0,0,0);
        glVertex3f(-w,h,z);

        glVertex3f(w,h,z);
        glVertex3f(w,-h,z);

        glVertex3f(-w,h,z);
        glVertex3f(-w,-h,z);

        glVertex3f(-w,h,z);
        glVertex3f(w,h,z);

        glVertex3f(-w,-h,z);
        glVertex3f(w,-h,z);
        glEnd();

        glPopMatrix();

        // draw lines
        if(draw_graph){
            //Eigen::Vector3d current_frame = - pKF->GetWorldCoord();
            Eigen::Vector3d current_frame(Twc.at<float>(0,3), Twc.at<float>(1,3), Twc.at<float>(2,3));
            if(i > 0 ){
                glLineWidth(graph_line_width);
                glColor4f(0.0f,1.0f,0.0f,0.8f);
                glBegin(GL_LINES);
                glVertex3f(last_frame(0),last_frame(1),last_frame(2));
                glVertex3f(current_frame(0),current_frame(1),current_frame(2));
                glEnd();
            }
            last_frame = current_frame;
        }
    }
}

void Viewer::DrawCube(const float &size,const float x, const float y, const float z)
{
    pangolin::OpenGlMatrix M = pangolin::OpenGlMatrix::Translate(-x,-size-y,-z);
    glPushMatrix();
    M.Multiply();
    pangolin::glDrawColouredCube(-size,size);
    glPopMatrix();
}

void Viewer::DrawFPS(cv::Mat &image)
{
    int fps_local;
    {
        std::unique_lock<std::mutex> lock(m_fps);
        fps_local = fps;
    }
    cv::Point pt_text = cv::Point(5, 20);
    cv::putText(image, "FPS:"+std::to_string(fps_local), pt_text, cv::FONT_HERSHEY_SIMPLEX, 0.8, cv::Scalar(255, 50, 50), 2, CV_AA);
}

void Viewer::DrawCube(cv::Mat &image, Eigen::Matrix4d &Tcw)
{

    std::vector<Eigen::Vector3d> boxConers;
    boxConers.push_back(Eigen::Vector3d(size_cube_half, -size_cube_half, -size_cube));
    boxConers.push_back(Eigen::Vector3d(size_cube_half, size_cube_half, -size_cube));
    boxConers.push_back(Eigen::Vector3d(-size_cube_half, size_cube_half, -size_cube));
    boxConers.push_back(Eigen::Vector3d(-size_cube_half, -size_cube_half, -size_cube));

    boxConers.push_back(Eigen::Vector3d(size_cube_half, -size_cube_half, 0));
    boxConers.push_back(Eigen::Vector3d(size_cube_half, size_cube_half, 0));
    boxConers.push_back(Eigen::Vector3d(-size_cube_half, size_cube_half, 0));
    boxConers.push_back(Eigen::Vector3d(-size_cube_half, -size_cube_half, 0));

    Eigen::Matrix3d rotation = Tcw.block(0,0,3,3);
    Eigen::Vector3d translation(Tcw(0,3), Tcw(1,3), Tcw(2,3));

    int min_depth_index = -1;
    double min_distance = 100000;
    cv::Point* p = new cv::Point[8];
    // project box points into image frame
    for(int i = 0; i < 8 ; i ++){
        Eigen::Vector3d projected_point = intrinsic*(rotation * boxConers[i] + translation);
        p[i].x = projected_point(0) / projected_point(2);
        p[i].y = projected_point(1) / projected_point(2);
        if(projected_point(2) < min_distance){
            min_distance = projected_point(2);
            min_depth_index = i;
        }
        //std::cout << p[i].x << " " << p[i].y << "\n";
    }

    //first draw large depth plane
    /*
    int point_group[8][12] = {
        {0,4,5,1, 0,3,7,4, 0,1,2,3},
        {1,5,6,2, 1,0,4,5, 1,2,3,0},
        {2,1,5,6, 2,6,7,3, 2,3,0,1},
        {3,0,1,2, 3,7,4,0, 3,2,6,7},
        {4,5,1,0, 4,7,6,5, 4,0,3,7},
        {5,4,7,6, 5,6,2,1, 5,1,0,4},
        {6,2,1,5, 6,7,3,2, 6,5,4,7},
        {7,6,5,4, 7,3,2,6, 7,4,0,3}};
    */
    int point_group[8][12] = {
        {0,4,5,1, 0,3,7,4, 0,1,2,3},
        {1,0,4,5, 1,5,6,2, 1,2,3,0},
        {2,6,7,3, 2,1,5,6, 2,3,0,1},
        {3,2,6,7, 3,7,4,0, 3,0,1,2},
        {4,5,1,0, 4,0,3,7, 4,7,6,5},
        {5,1,0,4, 5,6,2,1, 5,4,7,6},
        {6,7,3,2, 6,2,1,5, 6,5,4,7},
        {7,3,2,6, 7,4,0,3, 7,6,5,4}};

    int npts[1] = {4};
    cv::Point plain[1][4];
    const cv::Point* ppt[1] = {plain[0]};

    plain[0][0] = p[point_group[min_depth_index][0]];
    plain[0][1] = p[point_group[min_depth_index][1]];
    plain[0][2] = p[point_group[min_depth_index][2]];
    plain[0][3] = p[point_group[min_depth_index][3]];
    cv::fillPoly(image, ppt, npts, 1, cv::Scalar(200, 0, 0));

    plain[0][0] = p[point_group[min_depth_index][4]];
    plain[0][1] = p[point_group[min_depth_index][5]];
    plain[0][2] = p[point_group[min_depth_index][6]];
    plain[0][3] = p[point_group[min_depth_index][7]];
    cv::fillPoly(image, ppt, npts, 1, cv::Scalar(0, 200, 0));

    plain[0][0] = p[point_group[min_depth_index][8]];
    plain[0][1] = p[point_group[min_depth_index][9]];
    plain[0][2] = p[point_group[min_depth_index][10]];
    plain[0][3] = p[point_group[min_depth_index][11]];
    cv::fillPoly(image, ppt, npts, 1, cv::Scalar(0, 0, 200));
}

} // namespace
