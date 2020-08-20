#ifndef BASTIAN_FACTOR_H
#define BASTIAN_FACTOR_H


#include <eigen3/Eigen/Dense>
#include <ceres/ceres.h>

#include "PoseLyParameterization.h"
#include "GKeyFrame.h"

namespace BASTIAN
{

class PinholeCamera;
// template<int kNumResiduals, int N0 = 0, int N1 = 0, int N2 = 0, int N3 = 0, int N4 = 0,
//                             int N5 = 0, int N6 = 0, int N7 = 0, int N8 = 0, int N9 = 0>
// our number of residual should be 2 (the distance in 2D image space)
//     -> reproject the map point into image frame
// we only optimize one node, which is the camera pose, whose dimension is 7
class ProjectionCameraMapFactorPoseOnly : public ceres::SizedCostFunction<2, 7>
{
public:
    ProjectionCameraMapFactorPoseOnly(const Eigen::Vector3d &_map_point, const Eigen::Vector2d &_camera_point,
                           PinholeCamera *pPinholeCamera_);

    // calculate jacobian and residuals here
    //    -> residual should be Vector2d
    //    -> jacobian should be 1 * (2 * 7) -> ( one node * ( 2 residual elements * node's dimension 7 ))
    virtual bool Evaluate(double const *const *parameters, double *residuals, double **jacobians) const;
    void check(double **parameters);

private:
    Eigen::Vector3d map_point;
    Eigen::Vector2d camera_point;
    PinholeCamera *pPinholeCamera;
    double fx, fy, cx, cy;
};


// our number of residual should be 2 (the distance in 2D image space)
//     -> reproject the map point into image frame
// we optimize two nodes, the camera pose, whose dimension is 7
//                        and map point position, whose dimension is 3
class ProjectionCameraMapFactor : public ceres::SizedCostFunction<2, 7, 3>
{
public:
    ProjectionCameraMapFactor(const Eigen::Vector2d &_camera_point,
                           PinholeCamera *pPinholeCamera_);

    // calculate jacobian and residuals here
    //    -> residual should be Vector2d
    //    -> jacobian should be (2 * 7) and (2 * 3)
    virtual bool Evaluate(double const *const *parameters, double *residuals, double **jacobians) const;

    double EvaluateMine(double* camera_pose, double* map_point);

    //void check(double **parameters);

private:
    Eigen::Vector2d camera_point;
    PinholeCamera *pPinholeCamera;
    double fx, fy, cx, cy;
};

} // namespace
#endif // BASTIAN_FACTOR_H
