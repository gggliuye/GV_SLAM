#include "Map.h"

namespace BASTIAN
{

GMap::GMap()
{
}

void GMap::ReInitMapWithMarkerPoints(GKeyFrame *gkeyframe)
{
    {
        std::unique_lock<std::mutex> lock(m_keyframes_all);
        vKeyFramesAll.clear();
        vKeyFramesAll.push_back(gkeyframe);
    }

    {
        std::unique_lock<std::mutex> lock(m_map_points_all);
        vMapPointsAll.clear();
        std::vector<GMapPoint*> keyframeMapPints = gkeyframe->GetMapPoints();
        //vMapPointsAll.reserve(keyframeMapPints.size());
        for(size_t i = 0 ; i < keyframeMapPints.size() ; i ++){
            GMapPoint* pMp = keyframeMapPints[i];
            if(pMp)
                vMapPointsAll.insert(pMp);
        }
    }
}

void GMap::AddNewKeyFrame(GKeyFrame* pGKeyFrame)
{
    {
        std::unique_lock<std::mutex> lock(m_keyframes_all);
        vKeyFramesAll.push_back(pGKeyFrame);
    }
    {
        std::unique_lock<std::mutex> lock(m_map_points_all);
        std::vector<GMapPoint*> keyframeMapPints = pGKeyFrame->GetMapPoints();
        for(size_t i = 0 ; i < keyframeMapPints.size() ; i ++){
            GMapPoint* pMp = keyframeMapPints[i];
            if(pMp && pMp->GetbTriangulated())
                vMapPointsAll.insert(pMp);
        }
    }
    SetTodoLocalMapping(true);
    //DetachLocalMapping();
}

void GMap::SetTodoLocalMapping(bool b_in)
{
    std::unique_lock<std::mutex> lock(m_todo_localmapping);
    todo_localmapping = b_in;
}

bool GMap::GetTodoLocalMapping()
{
    std::unique_lock<std::mutex> lock(m_todo_localmapping);
    return todo_localmapping;
}

void GMap::RunLocalMapping()
{
    while(true){
        if(GetTodoLocalMapping()){
            CERES_OPTIMIZATION::OptimizeLocalMapRANSAC(this);
            CERES_OPTIMIZATION::OptimizeLocalMapRANSAC(this, true);
            CERES_OPTIMIZATION::OptimizeLocalMapRANSAC(this);
            //CERES_OPTIMIZATION::OptimizeLocalMapRANSAC(this);
            SetTodoLocalMapping(false);
            std::this_thread::sleep_for(std::chrono::microseconds(10));
        } else {
            std::this_thread::sleep_for(std::chrono::microseconds(100));
        }
    }
}


bool GMap::DetachLocalMapping()
{
    {
        std::unique_lock<std::mutex> lock(m_isMapping);
        if(isMapping){
            std::cout << "Local mapping is already working !\n";
            return false;
        }
    }

    std::thread *pThreadLocalMapping = new std::thread(&GMap::LocalMapping, this);
    pThreadLocalMapping->detach();
    std::cout << " Local mapping detached.\n";
    // if the main process stopped before will have problem
    return true;
}

void GMap::LocalMapping()
{
    {
        std::unique_lock<std::mutex> lock(m_isMapping);
        isMapping = true;
    }
    //std::cout << " Start local mapping.\n";

    CERES_OPTIMIZATION::OptimizeLocalMap(this);

    //std::cout << " End local mapping.\n\n";
    {
        std::unique_lock<std::mutex> lock(m_isMapping);
        isMapping = false;
    }
}

std::vector<GMapPoint*> GMap::GetAllMapPoints()
{
    std::unique_lock<std::mutex> lock(m_map_points_all);
    return std::vector<GMapPoint*>(vMapPointsAll.begin(), vMapPointsAll.end());
}

std::vector<GKeyFrame*> GMap::GetAllKeyFrames()
{
    std::unique_lock<std::mutex> lock(m_keyframes_all);
    return std::vector<GKeyFrame*>(vKeyFramesAll.begin(), vKeyFramesAll.end());
}

std::vector<GKeyFrame*> GMap::GetRecentKeyFrames(int num)
{
    std::unique_lock<std::mutex> lock(m_keyframes_all);
    int num_kf = vKeyFramesAll.size();
    if(num > num_kf){
        return std::vector<GKeyFrame*>(vKeyFramesAll.begin(), vKeyFramesAll.end());
    } else {
        return std::vector<GKeyFrame*>(vKeyFramesAll.end()-num, vKeyFramesAll.end());
    }
}

} // namespace BASTIAN
