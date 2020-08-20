#ifndef BASTIAN_MAP_H
#define BASTIAN_MAP_H

#include "thread"

#include "GKeyFrame.h"
#include "GMapPoint.h"

namespace BASTIAN
{

class GMapPoint;
class GKeyFrame;

class GMap
{
public:
    GMap();
    ~GMap(){}

    void ReInitMapWithMarkerPoints(GKeyFrame *gkeyframe);

    void AddNewKeyFrame(GKeyFrame* pGKeyFrame);

    std::vector<GMapPoint*> GetAllMapPoints();
    std::vector<GKeyFrame*> GetAllKeyFrames();
    std::vector<GKeyFrame*> GetRecentKeyFrames(int num);

private:
    std::mutex m_keyframes_all;
    std::vector<GKeyFrame*> vKeyFramesAll;

    std::mutex m_map_points_all;
    std::set<GMapPoint*> vMapPointsAll;

    // loop keyframes for global bundle adjustment use
    std::vector<GKeyFrame*> vKeyFramesLoop;

///// local mappint
public:
    void RunLocalMapping();
    bool DetachLocalMapping();
    void LocalMapping();

    bool GetTodoLocalMapping();
    void SetTodoLocalMapping(bool b_in);

private:
    std::mutex m_isMapping;
    bool isMapping = false;

    std::mutex m_todo_localmapping;
    bool todo_localmapping = false;

};

} // namespace BASTIAN


#endif // BASTIAN_MAP_H
