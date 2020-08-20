## Develop Log

# 1.0 Test different patch matching algorithm 
Testing different algorithms, including SSD, NCC version 1, NCC version 2,  etc. Finally find the current NCC version has the best performance.

# 2.0 Test openmp multi-thread processing

If using multi-core to run the application : boost from 50FPS(one core CPU) to 100FPS(4 core CPU).
 - not all the process are boosted, some of them actually has inverse effect.

And most of mine tests and further usage will be based on one processor, as a result, I do not implement the openmp multi-thread processing in the application.

# 3.0 Test pyramid brute matching processing

Consider the patch match process is similar to 2D lidar SLAM processing (which has about 3 degree of freedoms, : x\ y two translation and one rotation, and in a system with not much DoF, a brute method have been proven to be extremely efficient, seen Cartographer for an example).

As a result, I choose to use a pyramid brute patch matching method to detect marker image.

* resize to small initial size, and use gaussian filter to blur the image.
* build a multi-level pyramid marker image system.
* use brute matching to find corresonding matches in the input frame.


## performance

i7 CPU processor one core processing
