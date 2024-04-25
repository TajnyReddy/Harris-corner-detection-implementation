# Harris Corner Detector
### Introduction
This repository contains a Python implementation of the Harris Corner Detector algorithm. The Harris Corner Detector is a popular method used in computer vision for detecting corners in images.
![image](https://github.com/TajnyReddy/Harris-corner-detection-implementation/assets/59600478/80b659f9-c33d-41d1-9fab-78bf29adb368)

### Step-by-step solution:
* Convert to Grayscale: Convert the image to grayscale.
* Calculate Image Derivatives: Compute the image derivatives using Sobel operators.
* Compute Harris Matrix (M): Calculate the Harris matrix for each pixel.
* Harris Response Calculation: Calculate a corner response value for each pixel.
* Thresholding: Apply a threshold to the corner response values.
* Corner Detection: Identify pixels with high corner response values as corners.
* Highlight Corners: Draw blue circles around detected corners on the original image.
