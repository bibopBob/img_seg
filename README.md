# Image Segmentation
Simple image segmentation using computer vision. Python packages used along are NumPy, Matplotlib, and OpenCV.

Is provided a small dataset of images of clownfish to demonstrate the color space segmentation technique. Clownfish are easily identifiable by their bright orange color.

[See this repo Demo](https://img-seg-1782.herokuapp.com/)


## 1. Add Buildpacks (settings)
Heroku Buildpacks are sets of open source scripts that are used for compiling apps on Heroku.

## 2. Aptfile
This file is only needed if the application to be deployed uses opencv.

This file should be named as 'Aptfile' and should not have any extension.

It is required since opencv has some additional dependencies that are not installed on heroku by default. The names of those dependencies helps heroku to install these dependencies using the apt buildpack.

~~~
libsm6
libxrender1
libfontconfig1
libice6
~~~


## 3. requirements.txt
Use opencv-python-headless as it is out of libSM6 dependency.
~~~
opencv-python-headless==4.2.0.32
~~~
