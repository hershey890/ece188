Stage 1
=======
STAGE 1 GOALPOSTS
-----------------
PSNR/SSIM Goalposts
Gaussian blur: 26.5/0.65
Motion blur: 27.5/0.70
Salt and Pepper noise: 26.5/0.90
Gaussian noise: 19.5/0.60
Speckle noise: 20.0/0.65


Stage 2 
=======


Stage 3
=======
- https://www.cvlibs.net/datasets/kitti/eval_object.php?obj_benchmark=3d
    - download the datasets for left color images, velodyne point clouds, and camera calibration matrices
    - save them all in the following structure:
    data/
        kitti/
            object/
                calib
                    000000.txt
                image_2
                    000000.png
                label_2
                    000000.txt
                velodyne
                    000000.bin
                pred
                    000000.txt
            training
                calib
                    000000.txt
                image_2
                    000000.png
                label_2
                    000000.txt
                velodyne
                    000000.bin
                pred # you can ignore this one, its just for visualizing your predictions after you train the neural net
                    000000.txt

- https://github.com/kuixu/kitti_object_vis/tree/master visualizes lidar files
    - if you try this on some weird linux terminal lacking graphics support it's probably not going to work. The jupyter notebook may work though, I just haven't tried it
    - to setup the program, setup the conda environment as the README on the github says, but instead of installing everything with conda just install it with pip
    - conda install -c conda-forge pyqt
    - edit kitti_object_vis/kitti_object.py:
        - add a "from mayavi import mlab" line at the top below the "import cv2" line
        - comment out every other instance of "import mayavi.mlab as mlab"
    - to run the program, run the command "python kitti_object.py --show_lidar_with_depth --img_fov --const_box --vis"
    - you may need to run it as "python3 kitti_object.py -d /data//kitti/object/ --show_lidar_with_depth --img_fov --const_box --vis"

- https://github.com/koraykoca/Kitti2TFrecords converts kitti data to tfrecords