#include <iostream>

#include<opencv2/opencv.hpp>
#include<opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>

#include "GKeyFrame.h"
#include "MarkerFinder.h"
#include "Test_Cpp.h"
#include "System.h"

using namespace BASTIAN;
using namespace std;

string dataPath = "/home/yennefer/UTOPA/configurations/maker_images/popORB.dat";

int main(int argc, char **argv)
{
    cout << " Tests for learning C++ : \n\n";

    cout << " 1. first test -> vector copy :\n";
    BASTIAN_TEST::TestVectorCopySpeed();
    cout << "    equal operater = and create with begin() end() are all copy\n";
    cout << "    operations, and the second one is slightly faster.\n\n";

    cout << " 2. second test -> bundle adjustment :\n";
    BASTIAN_TEST::TestOptimization();
    cout << "    mine optimization process works just fine.\n\n";

    cout << " 3. thrid test -> the creation and deletion of pointers :\n";
    BASTIAN_TEST::TestPointerDeletion();

    cout << " 4. test triangualtion : \n";
    BASTIAN_TEST::TestTwoViewTrangulation();

    cout << " 5. second test -> bundle adjustment 2.0 :\n";
    BASTIAN_TEST::TestOptimizationWithPoints();

    return 0;
}
