#root/components/CMakeLists.txt

ssig_add_module(core REQUIRED OPENCV opencv_world opencv_core opencv_imgproc opencv_imgcodecs opencv_highgui DEPENDENCIES libflann)
ssig_add_module(ml OPENCV opencv_core opencv_ml opencv_imgproc opencv_objdetect opencv_highgui opencv_world DEPENDENCIES core libsvm libflann)
ssig_add_module(video OPENCV opencv_world opencv_core opencv_videoio opencv_video DEPENDENCIES core libflann)
ssig_add_module(descriptors CUDA OPENCV opencv_core opencv_ml opencv_imgproc opencv_objdetect opencv_highgui opencv_videoio opencv_video opencv_world DEPENDENCIES core video libflann)
ssig_add_module(hashing OPENCV opencv_core DEPENDENCIES core ml libflann)
#ssf_add_module(optimization OPENCV opencv_core DEPENDENCIES core)
