#include "Test_Cpp.h"

namespace BASTIAN_TEST
{

    void TestVectorCopy()
    {
        // Initializing vector with values
        vector<int> vect1{1, 2, 3, 4};
        // Declaring new vector
        vector<int> vect2, vect3;
        // Using assignment operator to copy one
        // vector to other
        vect2 = vect1;
        vect3 = vector<int>(vect1.begin(), vect1.end());
        cout << "Old vector elements are : ";
        for (size_t i=0; i<vect1.size(); i++)
            cout << vect1[i] << " ";
        cout << endl;
        cout << "New vector 1 elements are : ";
        for (size_t i=0; i<vect2.size(); i++)
            cout << vect2[i] << " ";
        cout<< endl;
        cout << "New vector 2 elements are : ";
        for (size_t i=0; i<vect3.size(); i++)
            cout << vect3[i] << " ";
        cout<< endl;
        // Changing value of vector to show that a new
        // copy is created.
        vect1[0] = 2;
        cout << "The first element of old vector is :";
        cout << vect1[0] << endl;
        cout << "The first element of new vector 1 is :";
        cout << vect2[0] <<endl;
        cout << "The first element of new vector 2 is :";
        cout << vect3[0] <<endl;
    }

    void TestVectorCopySpeed()
    {
        // Initializing vector with values
        vector<int> vect1;
        int num = 1000000;
        vect1.reserve(num);
        for(int i = 0; i < num ; i ++){
            vect1.push_back(i);
        }

        // Declaring new vector
        vector<int> vect2, vect3, vect4;

        TicToc time1;
        vect2 = vect1;
        std::cout << "Copy vector with = takes : " << time1.toc() << "\n";

        TicToc time2;
        vect3 = vector<int>(vect1.begin(), vect1.end());
        std::cout << "Copy vector with begin() end() takes : " << time2.toc() << "\n";

        //vect4 = vector<int>(vect1.end()-21, vect1.end());
        //std::cout << vect4.size() << std::endl;
        // results:
        //Copy vector with = takes : 1.65398
        //Copy vector with begin() end() takes : 1.64626

        // as a result, I should better use "begin() end()" to copy, since it is
        // slightly faster
    }

    /* 2020/01/09
    * In a lots of process, we need to access the values of the class
    *      1. set these values as public. (need mutex and not safe)
    *   X  2. get a copy of the values. (most of the vectors are vector of pointers,
    *               copying is fast, and more important, it is safe)
    */
    void TestOptimization()
    {
        double camera_pose[7] = {0.1, -0.2, 0.4, 0.02, 0.04, -0.02, 0.99};
        BASTIAN::PinholeCamera *pPinholeCamera = new BASTIAN::PinholeCamera(500, 500, 320, 240, 640, 480);

        Eigen::Matrix3d intrinstic;
        intrinstic << pPinholeCamera->fx, 0 , pPinholeCamera->cx,
                      0, pPinholeCamera->fy, pPinholeCamera->cy,
                      0, 0, 1;

        Eigen::Vector3d Pc(camera_pose[0], camera_pose[1], camera_pose[2]);
        Eigen::Quaterniond Qc(camera_pose[6], camera_pose[3], camera_pose[4], camera_pose[5]);
        Qc.normalize();

        int num = 20;
        std::vector<cv::Point3f> matchedMapPoints;
        std::vector<cv::Point2f> matchedImagePoints;
        matchedMapPoints.reserve(num);
        matchedImagePoints.reserve(num);

        double w_sigma = 0.2;
        std::default_random_engine generator;
        std::normal_distribution<double> noise(0.,w_sigma);

        double simulateRadius = 5;
        std::default_random_engine generatorP;
        std::normal_distribution<double> pointP(0.,simulateRadius);

        for(int i = 0 ; i < num ; i ++)
        {
            Eigen::Vector3d map_point;
            map_point << pointP(generator), pointP(generator), pointP(generator);

            Eigen::Vector3d pixel_point_camera = intrinstic * (Qc * map_point + Pc);

            cv::Point2f camera_point;
            camera_point.x = pixel_point_camera(0) / pixel_point_camera(2) + noise(generator);
            camera_point.y = pixel_point_camera(1) / pixel_point_camera(2) + noise(generator);

            cv::Point3f map_point_cv(map_point(0), map_point(1), map_point(2));

            matchedMapPoints.push_back(map_point_cv);
            matchedImagePoints.push_back(camera_point);
        }

        double pose[7] = {0.0, -0.1, 0.2, 0, 0, 0, 1};
        BASTIAN::GKeyFrame* gKF = new BASTIAN::GKeyFrame(pPinholeCamera, matchedMapPoints,
                                    matchedImagePoints);
        gKF->SetPoseTcwArray(pose);
        BASTIAN::CERES_OPTIMIZATION::OptimizeCameraPose(gKF);

        double result[7];
        gKF->GetPoseTcwArray(result);

        std::cout << "Ground Truth : ";
        for(int i = 0 ; i < 7 ; i++)
            std::cout << camera_pose[i] << " ";

        std::cout << "\nInitial Pose : ";
        for(int i = 0 ; i < 7 ; i++)
            std::cout << pose[i] << " ";

        std::cout << "\nFinal Result : ";
        for(int i = 0 ; i < 7 ; i++)
            std::cout << result[i] << " ";
        std::cout << std::endl;
        //0.626828 -0.304482 0.249544 -0.0180825 -0.0250154 -0.00734673 0.999497
    }

    void TestOptimizationWithPoints()
    {
        double camera_pose[7] = {0.1, -0.2, 0.4, 0.02, 0.04, -0.02, 0.99};
        BASTIAN::PinholeCamera *pPinholeCamera = new BASTIAN::PinholeCamera(500, 500, 320, 240, 640, 480);

        Eigen::Matrix3d intrinstic;
        intrinstic << pPinholeCamera->fx, 0 , pPinholeCamera->cx,
                      0, pPinholeCamera->fy, pPinholeCamera->cy,
                      0, 0, 1;

        Eigen::Vector3d Pc(camera_pose[0], camera_pose[1], camera_pose[2]);
        Eigen::Quaterniond Qc(camera_pose[6], camera_pose[3], camera_pose[4], camera_pose[5]);
        Qc.normalize();

        int num = 10;
        std::vector<cv::Point3f> matchedMapPoints;
        std::vector<cv::Point2f> matchedImagePoints;
        matchedMapPoints.reserve(num);
        matchedImagePoints.reserve(num);

        double w_sigma = 0.2;
        std::default_random_engine generator;
        std::normal_distribution<double> noise(0.,w_sigma);

        double w_sigma_m = 0.01;
        std::default_random_engine generatorM;
        std::normal_distribution<double> noise_m(0.,w_sigma_m);

        double simulateRadius = 5;
        std::default_random_engine generatorP;
        std::normal_distribution<double> pointP(0.,simulateRadius);

        for(int i = 0 ; i < num ; i ++)
        {
            Eigen::Vector3d map_point;
            map_point << pointP(generator), pointP(generator), pointP(generator);

            Eigen::Vector3d map_point_noise(noise_m(generatorM),noise_m(generatorM), noise_m(generatorM));

            Eigen::Vector3d pixel_point_camera = intrinstic * (Qc * map_point + Pc);

            cv::Point2f camera_point;
            camera_point.x = pixel_point_camera(0) / pixel_point_camera(2) + noise(generator);
            camera_point.y = pixel_point_camera(1) / pixel_point_camera(2) + noise(generator);

            map_point = map_point + map_point_noise;
            cv::Point3f map_point_cv(map_point(0), map_point(1), map_point(2));

            matchedMapPoints.push_back(map_point_cv);
            matchedImagePoints.push_back(camera_point);
        }

        double pose[7] = {0.08, -0.15, 0.35, 0, 0, 0, 1};
        BASTIAN::GKeyFrame* gKF = new BASTIAN::GKeyFrame(pPinholeCamera, matchedMapPoints,
                                    matchedImagePoints, false);

        gKF->SetPoseTcwArray(pose);
        BASTIAN::CERES_OPTIMIZATION::OptimizeCameraPoseAndMapPoint(gKF);

        double result[7];
        gKF->GetPoseTcwArray(result);

        std::cout << "Ground Truth : ";
        for(int i = 0 ; i < 7 ; i++)
            std::cout << camera_pose[i] << " ";

        std::cout << "\nInitial Pose : ";
        for(int i = 0 ; i < 7 ; i++)
            std::cout << pose[i] << " ";

        std::cout << "\nFinal Result : ";
        for(int i = 0 ; i < 7 ; i++)
            std::cout << result[i] << " ";
        std::cout << std::endl;
        std::cout << " need to fix some of the points. this test is useless, ";
        std::cout << " but in real project, the function is proven correct. \n\n";
    }

    void TestPointerDeletion()
    {
        int num = 10;
        std::vector<TestPointer*> vTestPointer1;
        std::vector<TestPointer*> vTestPointer2;
        vTestPointer1.reserve(num);
        vTestPointer2.reserve(num);
        for(int i = 0 ; i < num; i++){
            TestPointer* pTestPointer = new TestPointer(i);
            vTestPointer1.push_back(pTestPointer);
            vTestPointer2.push_back(pTestPointer);
        }
        std::cout << " delete the fifth element from vector1 : \n";
        TestPointer* pTestPointer = vTestPointer1[5];
        delete pTestPointer;

        std::cout << " test the fifth element in vector1 : \n";
        if(vTestPointer1[5]){
            std::cout << "  the count number is : " << vTestPointer1[5]->count << "\n";
        } else {
            std::cout << "  this elemetn is null. \n";
        }

        std::cout << " test the fifth element in vector2 : \n";
        if(vTestPointer2[5]){
            std::cout << "  the count number is : " << vTestPointer2[5]->count << "\n";
        } else {
            std::cout << "  this elemetn is null. \n";
        }

        TestPointer* pTestPointert = new TestPointer(99);
        std::cout << " create a new pointer with the count "<< pTestPointert->count <<" , which will use the upper \n";
        std::cout << " deleted empty memory automaticly. \n";
        std::cout << " test the fifth element in vector2 : \n";
        if(vTestPointer2[5]){
            std::cout << "  the count number is : " << vTestPointer2[5]->count << "\n";
        } else {
            std::cout << "  this elemetn is null. \n";
        }

        std::cout << "  the pointer is truly deleted, but we also need to set the element\n";
        std::cout << "  to be NULL, otherwise, it will keep pointing to the same adress\n";
        std::cout << "  which will leads to error.  -> pointer is only a value of address.\n\n";

    }

    void MakeTestDataForTrangulation(const Eigen::Vector3d euler_angle, const Eigen::Vector3d translation,
                                    const Eigen::Vector3d point_ground_truth,
                                    Eigen::Matrix4d &Tcw, Eigen::Vector2d &p_homo)
    {
        // camera pose Twc
        Eigen::AngleAxisd rollAngle(Eigen::AngleAxisd(euler_angle(2),Eigen::Vector3d::UnitX()));
        Eigen::AngleAxisd pitchAngle(Eigen::AngleAxisd(euler_angle(1),Eigen::Vector3d::UnitY()));
        Eigen::AngleAxisd yawAngle(Eigen::AngleAxisd(euler_angle(0),Eigen::Vector3d::UnitZ()));
        Eigen::Matrix3d rotation_matrix = (yawAngle * pitchAngle * rollAngle).matrix();
        // Tcw
        //Eigen::Matrix4d Tcw;
        Tcw.setIdentity();
        Tcw.block(0,0,3,3) = rotation_matrix.transpose();
        Eigen::Vector3d translation_inv = - rotation_matrix.transpose() * translation;
        Tcw.block(0,3,3,1) = translation_inv;

        // get pixel point
        Eigen::Vector3d point_camera = rotation_matrix.transpose() * point_ground_truth + translation_inv;
        Eigen::Vector2d pt_homo(point_camera(0)/point_camera(2), point_camera(1)/point_camera(2));
        p_homo = pt_homo;
    }

    void TestTwoViewTrangulation()
    {
        Eigen::Vector3d point_ground_truth(0.18, -0.1, 0.5);

        double fx = 600;
        double fy = 600;
        double cx = 320;
        double cy = 240;

        // noise of observaton (pixel level)
        double w_sigma = 0.001;
        std::default_random_engine generator;
        std::normal_distribution<double> noise(0.,w_sigma);

        Eigen::Vector3d euler_angle_1(0, 0, 0);
        Eigen::Vector3d translation_1(-0.1, 0.0, 0.0);
        Eigen::Matrix4d Tcw_1;
        Eigen::Vector2d pt_homo_1;
        MakeTestDataForTrangulation(euler_angle_1, translation_1,point_ground_truth,Tcw_1, pt_homo_1);
        Eigen::Vector2d noise_1(noise(generator), noise(generator));
        pt_homo_1 = pt_homo_1 + noise_1;
        BASTIAN::GMapPoint *testPoint = new BASTIAN::GMapPoint(Tcw_1, pt_homo_1);
        std::cout << " the first camera view is : \n" << Tcw_1 << "\n";
        std::cout << " camera homo pixel is : " << pt_homo_1.transpose() << "\n";
        Eigen::Vector2d camera_pixel_1(pt_homo_1(0)*fx+cx,pt_homo_1(1)*fy+cy);
        std::cout << " camera pixel is : " << camera_pixel_1.transpose() << "\n";

        Eigen::Vector3d euler_angle_2(0, 0, 0);
        Eigen::Vector3d translation_2(0.1, 0.0, 0.0);
        Eigen::Matrix4d Tcw_2;
        Eigen::Vector2d pt_homo_2;
        MakeTestDataForTrangulation(euler_angle_2, translation_2,point_ground_truth,Tcw_2, pt_homo_2);
        Eigen::Vector2d noise_2(noise(generator), noise(generator));
        pt_homo_2 = pt_homo_2 + noise_2;
        std::cout << " the second camera view is : \n" << Tcw_2 << "\n";
        std::cout << " camera homo pixel is : " << pt_homo_2.transpose() << "\n";
        Eigen::Vector2d camera_pixel_2(pt_homo_2(0)*fx+cx,pt_homo_2(1)*fy+cy);
        std::cout << " camera pixel is : " << camera_pixel_2.transpose() << "\n";

        std::cout << "   ground truth point is : " << point_ground_truth.transpose() << "\n";
        if(testPoint->Triangulate(Tcw_2, pt_homo_2)){
            std::cout << " triangulation result is : " << testPoint->GetPose().transpose() << "\n\n";
        } else {
            std::cout << " Triangulation failed. \n";
            std::cout << " triangulation result is : " << testPoint->GetPose().transpose() << "\n\n";
        }

    }


} // namespace BASTIAN_TEST
