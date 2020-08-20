#include "Optimization.h"

namespace BASTIAN
{

int CERES_OPTIMIZATION::OptimizeCameraPose(GKeyFrame* pGKeyFrame)
{
    bool use_robust_fcn = true;
    bool check_jacobian = false;
    int tracked_count = 0;

    double camera_pose[7];
    pGKeyFrame->GetPoseTcwArray(camera_pose);

    ceres::Problem problem;
    ceres::LossFunctionWrapper* loss_function = new ceres::LossFunctionWrapper(new ceres::CauchyLoss(1.0), ceres::TAKE_OWNERSHIP);

    ceres::LocalParameterization *local_parameterization = new PoseLyParameterization();
    problem.AddParameterBlock(camera_pose, 7, local_parameterization);

    std::vector<GMapPoint*> keyframeMapPints = pGKeyFrame->GetMapPoints();
    for(size_t i = 0, iend = keyframeMapPints.size() ; i < iend ; i++){
        GMapPoint* pMP = keyframeMapPints[i];
        if(!pMP)
            continue;
        if(!pMP->GetbTriangulated() || pMP->GetBadFlag())
            continue;

        tracked_count++;
        const cv::Point2f &kp = pGKeyFrame->vTrackedPoints[i];
        Eigen::Vector2d camera_point(kp.x, kp.y);
        Eigen::Vector3d map_point = pMP->GetPose();

        ProjectionCameraMapFactorPoseOnly *f_camera_pose = new ProjectionCameraMapFactorPoseOnly(map_point, camera_point,
                         pGKeyFrame->pPinholeCamera);

        if(check_jacobian){
            double **jaco = new double *[1];
            jaco[0] = camera_pose;
            f_camera_pose->check(jaco);
        }

        if(use_robust_fcn && !pMP->GetFixed())
            problem.AddResidualBlock(f_camera_pose, loss_function, camera_pose);
        else
            problem.AddResidualBlock(f_camera_pose, NULL, camera_pose);
    }

    ceres::Solver::Options options;
    options.linear_solver_type = ceres::DENSE_SCHUR;
    options.trust_region_strategy_type = ceres::DOGLEG;
    options.max_num_iterations = 20;
    options.minimizer_progress_to_stdout = false;

    ceres::Solver::Summary summary;
    ceres::Solve(options, &problem, &summary);
    //std::cout << summary.BriefReport() << std::endl;
    //printf("Iterations : %d \n", static_cast<int>(summary.iterations.size()));

    pGKeyFrame->SetPoseTcwArray(camera_pose);

    return tracked_count;
}


int CERES_OPTIMIZATION::OptimizeCameraPoseAndMapPoint(GKeyFrame* pGKeyFrame)
{
    bool use_robust_fcn = false;
    int tracked_count = 0;

    double camera_pose[7];
    pGKeyFrame->GetPoseTcwArray(camera_pose);

    ceres::Problem problem;
    ceres::LossFunctionWrapper* loss_function = new ceres::LossFunctionWrapper(new ceres::HuberLoss(1.0), ceres::TAKE_OWNERSHIP);

    ceres::LocalParameterization *local_parameterization = new PoseLyParameterization();
    problem.AddParameterBlock(camera_pose, 7, local_parameterization);

    std::vector<GMapPoint*> keyframeMapPints = pGKeyFrame->GetMapPoints();

    int num_feature = keyframeMapPints.size();
    //double map_poses[num_feature][3];
    //int id_count = 0;
    for(int i = 0; i < num_feature ; i++){
        GMapPoint* pMP = keyframeMapPints[i];
        if(!pMP)
            continue;
        if(!pMP->GetbTriangulated() || pMP->GetBadFlag())
            continue;

        tracked_count++;
        const cv::Point2f &kp = pGKeyFrame->vTrackedPoints[i];
        Eigen::Vector2d camera_point(kp.x, kp.y);
        Eigen::Vector3d map_point = pMP->GetPose();

        if(pMP->GetFixed()){
            ProjectionCameraMapFactorPoseOnly *f_camera_pose = new ProjectionCameraMapFactorPoseOnly(map_point, camera_point,
                             pGKeyFrame->pPinholeCamera);
            problem.AddResidualBlock(f_camera_pose, NULL, camera_pose);
        } else {
            /*
            pMP->frame_parameter_id = id_count;
            map_poses[id_count][0] = map_point[0];
            map_poses[id_count][1] = map_point[1];
            map_poses[id_count][2] = map_point[2];

            ProjectionCameraMapFactor *f_all = new ProjectionCameraMapFactor(camera_point,
                             pGKeyFrame->pPinholeCamera);
            if(use_robust_fcn){
                problem.AddResidualBlock(f_all, loss_function, camera_pose, map_poses[id_count]);
            } else {
                problem.AddResidualBlock(f_all, NULL, camera_pose, map_poses[id_count]);
            }
            id_count++;
            */
            //pose_frame_opt
            ProjectionCameraMapFactor *f_all = new ProjectionCameraMapFactor(camera_point,
                             pGKeyFrame->pPinholeCamera);
            pMP->pose_frame_opt[0] = map_point[0];
            pMP->pose_frame_opt[1] = map_point[1];
            pMP->pose_frame_opt[2] = map_point[2];
            if(use_robust_fcn){
                problem.AddResidualBlock(f_all, loss_function, camera_pose, pMP->pose_frame_opt);
            } else {
                problem.AddResidualBlock(f_all, NULL, camera_pose, pMP->pose_frame_opt);
            }
        }
    }

    ceres::Solver::Options options;
    options.linear_solver_type = ceres::DENSE_SCHUR;
    options.trust_region_strategy_type = ceres::DOGLEG;
    options.max_num_iterations = 20;
    options.minimizer_progress_to_stdout = false;

    ceres::Solver::Summary summary;
    ceres::Solve(options, &problem, &summary);
    //std::cout << summary.BriefReport() << std::endl;
    //printf("Iterations : %d \n", static_cast<int>(summary.iterations.size()));

    // set camera pose
    pGKeyFrame->SetPoseTcwArray(camera_pose);

    // set optimized map points
    for(int i = 0; i < num_feature ; i++){
        GMapPoint* pMP = keyframeMapPints[i];
        if(!pMP)
            continue;
        if(!pMP->GetbTriangulated() || pMP->GetBadFlag())
            continue;
        if(pMP->GetFixed())
            continue;
        /*
        int idx = pMP->frame_parameter_id;
        if(idx < 0)
            continue;  // for the fixed points

        pMP->frame_parameter_id = -1;
        Eigen::Vector3d map_pose(map_poses[idx][0], map_poses[idx][1], map_poses[idx][2]);
        pMP->SetPose(map_pose);
        */
        Eigen::Vector3d map_pose(pMP->pose_frame_opt[0], pMP->pose_frame_opt[1], pMP->pose_frame_opt[2]);
        pMP->SetPose(map_pose);
    }

    return tracked_count;
}


int CERES_OPTIMIZATION::OptimizeCameraPoseRANSAC(GKeyFrame* pGKeyFrame)
{
    return -1;
}

void CERES_OPTIMIZATION::OptimizeLocalMap(GMap* pGMap, bool fix_camera)
{
    std::vector<GKeyFrame*> vRecentKeyframes = pGMap->GetRecentKeyFrames(LOCAL_NUM);

    bool use_robust_fcn = true;

    ceres::Problem problem;
    ceres::LossFunctionWrapper* loss_function = new ceres::LossFunctionWrapper(new ceres::HuberLoss(1.0), ceres::TAKE_OWNERSHIP);

    // get the camera poses
    int num_keyframes = vRecentKeyframes.size();
    double camera_pose[num_keyframes][7];
    for(int i = 0 ; i < num_keyframes ; i++){
        vRecentKeyframes[i]->GetPoseTcwArray(camera_pose[i]);
        ceres::LocalParameterization *local_parameterization = new PoseLyParameterization();
        problem.AddParameterBlock(camera_pose[i], 7, local_parameterization);
        if(fix_camera){
            problem.SetParameterBlockConstant(camera_pose[i]);
        }
    }
    std::cout << " [Local mapping] done set " << num_keyframes << " cameras.\n";

    // add feature constrains
    for(int j = 0 ; j < num_keyframes ; j++){
        GKeyFrame *pGKeyFrame = vRecentKeyframes[j];

        std::vector<GMapPoint*> keyframeMapPints = pGKeyFrame->GetMapPoints();
        int num_feature = keyframeMapPints.size();
        //std::cout << "    camera " << j << " has " << num_feature << " features.\n";

        for(int i = 0; i < num_feature ; i++){
            GMapPoint* pMP = keyframeMapPints[i];

            if(!pMP)
                continue;
            if(!pMP->GetbTriangulated() || pMP->GetBadFlag())
                continue;

            const cv::Point2f &kp = pGKeyFrame->vTrackedPoints[i];
            Eigen::Vector2d camera_point(kp.x, kp.y);
            Eigen::Vector3d map_point = pMP->GetPose();

            //std::cout << i << " " << pMP->GetFixed() << " " << map_point.transpose() << " \n";

            if(pMP->GetFixed()){
                ProjectionCameraMapFactorPoseOnly *f_camera_pose = new ProjectionCameraMapFactorPoseOnly(map_point, camera_point,
                                 pGKeyFrame->pPinholeCamera);
                problem.AddResidualBlock(f_camera_pose, NULL, camera_pose[j]);
            } else {
                //pose_frame_opt
                ProjectionCameraMapFactor *f_all = new ProjectionCameraMapFactor(camera_point,
                                 pGKeyFrame->pPinholeCamera);
                pMP->pose_local_opt[0] = map_point[0];
                pMP->pose_local_opt[1] = map_point[1];
                pMP->pose_local_opt[2] = map_point[2];
                if(use_robust_fcn){
                    problem.AddResidualBlock(f_all, loss_function, camera_pose[j], pMP->pose_local_opt);
                } else {
                    problem.AddResidualBlock(f_all, NULL, camera_pose[j], pMP->pose_local_opt);
                }
            }
        }
    }
    //std::cout << " [Local mapping] done set keypoints.\n";

    ceres::Solver::Options options;
    options.linear_solver_type = ceres::DENSE_SCHUR;
    options.trust_region_strategy_type = ceres::DOGLEG;
    options.max_num_iterations = 40;
    options.minimizer_progress_to_stdout = false;

    ceres::Solver::Summary summary;
    ceres::Solve(options, &problem, &summary);

    std::cout << " - Local Optimization : \n";
    std::cout << summary.BriefReport() << std::endl;
    //printf("Iterations : %d \n", static_cast<int>(summary.iterations.size()));

    // set camera poses
    for(int i = 0 ; i < num_keyframes ; i++){
        vRecentKeyframes[i]->SetPoseTcwArray(camera_pose[i]);
    }

    // update map points
    for(int i = 0 ; i < num_keyframes ; i++){
        std::vector<GMapPoint*> keyframeMapPints = vRecentKeyframes[i]->GetMapPoints();
        int num_feature = keyframeMapPints.size();

        for(int i = 0; i < num_feature ; i++){
            GMapPoint* pMP = keyframeMapPints[i];

            if(!pMP)
                continue;
            if(!pMP->GetbTriangulated() || pMP->GetBadFlag())
                continue;
            if(pMP->GetFixed())
                continue;

            pMP->SetPose(pMP->pose_local_opt);
        }
    }
}


void CERES_OPTIMIZATION::OptimizeLocalMapRANSAC(GMap* pGMap, bool fix_camera)
{
    std::vector<GKeyFrame*> vRecentKeyframes = pGMap->GetRecentKeyFrames(LOCAL_NUM);

    ceres::Problem problem;
    //ceres::LossFunctionWrapper* loss_function = new ceres::LossFunctionWrapper(new ceres::HuberLoss(1.0), ceres::TAKE_OWNERSHIP);

    // get the camera poses
    int num_keyframes = vRecentKeyframes.size();
    double camera_pose[num_keyframes][7];
    for(int i = 0 ; i < num_keyframes ; i++){
        vRecentKeyframes[i]->GetPoseTcwArray(camera_pose[i]);
        ceres::LocalParameterization *local_parameterization = new PoseLyParameterization();
        problem.AddParameterBlock(camera_pose[i], 7, local_parameterization);
        if(fix_camera){
            problem.SetParameterBlockConstant(camera_pose[i]);
        }
    }
    std::cout << " [Local mapping] done set " << num_keyframes << " cameras.\n";

    int outlier_count = 0;
    // add feature constrains
    for(int j = 0 ; j < num_keyframes ; j++){
        GKeyFrame *pGKeyFrame = vRecentKeyframes[j];

        std::vector<GMapPoint*> keyframeMapPints = pGKeyFrame->GetMapPoints();
        int num_feature = keyframeMapPints.size();

        for(int i = 0; i < num_feature ; i++){
            GMapPoint* pMP = keyframeMapPints[i];

            if(!pMP)
                continue;
            if(!pMP->GetbTriangulated() || pMP->GetBadFlag())
                continue;

            const cv::Point2f &kp = pGKeyFrame->vTrackedPoints[i];
            Eigen::Vector2d camera_point(kp.x, kp.y);
            Eigen::Vector3d map_point = pMP->GetPose();

            if(pMP->GetFixed()){
                ProjectionCameraMapFactorPoseOnly *f_camera_pose = new ProjectionCameraMapFactorPoseOnly(map_point, camera_point,
                                 pGKeyFrame->pPinholeCamera);
                problem.AddResidualBlock(f_camera_pose, NULL, camera_pose[j]);
            } else {
                //pose_frame_opt
                ProjectionCameraMapFactor *f_all = new ProjectionCameraMapFactor(camera_point,
                                 pGKeyFrame->pPinholeCamera);
                pMP->pose_local_opt[0] = map_point[0];
                pMP->pose_local_opt[1] = map_point[1];
                pMP->pose_local_opt[2] = map_point[2];

                double res = f_all->EvaluateMine(camera_pose[j], pMP->pose_local_opt);
                if(res > RANSAC_LOSS_THRESHOLD){
                    //pMP->SetBadFlag();
                    outlier_count++;
                } else {
                    problem.AddResidualBlock(f_all, NULL, camera_pose[j], pMP->pose_local_opt);
                }
            }
        }
    }
    //std::cout << " [Local mapping] done set keypoints.\n";

    ceres::Solver::Options options;
    options.linear_solver_type = ceres::DENSE_SCHUR;
    options.trust_region_strategy_type = ceres::DOGLEG;
    options.max_num_iterations = 40;
    options.minimizer_progress_to_stdout = false;

    ceres::Solver::Summary summary;
    ceres::Solve(options, &problem, &summary);

    std::cout << " - Local Optimization ( has " << outlier_count << " outliers ) : \n";
    std::cout << summary.BriefReport() << std::endl;
    //printf("Iterations : %d \n", static_cast<int>(summary.iterations.size()));

    // set camera poses
    for(int i = 0 ; i < num_keyframes ; i++){
        vRecentKeyframes[i]->SetPoseTcwArray(camera_pose[i]);
    }

    // update map points
    for(int i = 0 ; i < num_keyframes ; i++){
        std::vector<GMapPoint*> keyframeMapPints = vRecentKeyframes[i]->GetMapPoints();
        int num_feature = keyframeMapPints.size();

        for(int i = 0; i < num_feature ; i++){
            GMapPoint* pMP = keyframeMapPints[i];

            if(!pMP)
                continue;
            if(!pMP->GetbTriangulated() || pMP->GetBadFlag())
                continue;
            if(pMP->GetFixed())
                continue;

            pMP->SetPose(pMP->pose_local_opt);
        }
    }
}



} // namespace
