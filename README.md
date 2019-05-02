# TreeRingCounter
This is a Python script counts tree rings from an image. The tree stump needs to be sanded and the rings need to be reasonably visible. 
The easiest way to install Python is through Anconda distribution https://www.anaconda.com/distribution/#download-section

This script uses OpenCV library (computer vision and machine learning software library). To install OpenCV with pip open command line and type: 
pip install opencv-contrib-python –upgrade
to install via Anaconda (recommended if you use Anaconda distribution) open the Anaconda command prompt and type: conda install -c conda-forge opencv (see here: https://anaconda.org/conda-forge/opencv for more options)

The script will let you choose an image file, open the image and wait for you to draw a line (you can redraw the line as many times as you want). The script will count and map the tree rings along this line. After the line is drawn click Esc and the script will count and map the rings along the line. 

## Exapmle:

[!alt text](https://github.com/ronimos/TreeRingCounter/blob/master/Examples/Tree%20rings%20counter2.png)
