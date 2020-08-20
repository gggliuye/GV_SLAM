#ifndef BASTIAN_TEST_CPP_H
#define  BASTIAN_TEST_CPP_H

#include <random>

#include "utils.hpp"
#include "Optimization.h"
#include "GKeyFrame.h"

//using namespace BASTIAN;
using namespace std;

namespace BASTIAN_TEST
{

    void TestVectorCopy();
    void TestVectorCopySpeed();
    void TestOptimization();
    void TestOptimizationWithPoints();

    class TestPointer
    {
    public:
        TestPointer(int i ){
            a = 0; b = 1; count = i;
        }
        int a;
        int b;
        int count;
    };
    void TestPointerDeletion();

    void TestTwoViewTrangulation();

} // namespace BASTIAN_TEST


#endif //  BASTIAN_TEST_CPP_H
