#ifndef BASTIAN_OPTIMIZATION_H
#define BASTIAN_OPTIMIZATION_H

#include "Factors.h"
#include "GKeyFrame.h"
#include "parameters.h"
#include "Map.h"

namespace BASTIAN
{
class GKeyFrame;
class GMap;

class CERES_OPTIMIZATION
{
public:

    static int OptimizeCameraPose(GKeyFrame* pGKeyFrame);

    static int OptimizeCameraPoseAndMapPoint(GKeyFrame* pGKeyFrame);

    static int OptimizeCameraPoseRANSAC(GKeyFrame* pGKeyFrame);

    static void OptimizeLocalMap(GMap* pGMap, bool fix_camera = false);

    static void OptimizeLocalMapRANSAC(GMap* pGMap, bool fix_camera = false);

};




} // namespace

#endif // BASTIAN_OPTIMIZATION_H
