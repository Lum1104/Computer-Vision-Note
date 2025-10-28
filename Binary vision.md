---
layout: default
title: Binary Vision
nav_order: 4
permalink: /binary-vision/
description: Thresholding, morphology, and connectivity with OpenCV snippets.
---

- Thresholding
- Threshold Detection
- Variations
- Mathematical Morphology
- Connectivity
- Objects of interest vs background

## Thresholding

- Distinct foreground and background
- How to determine the best threshold 

## Threshold Detection

- Manual Setting
- Changing lighting
- Need to determine automatically
- Techniques:
  - Image
  - Histogram
  - Probability

### Otsu Thresholding

- If it's not two normal distributions
- Minimizes the spread of the pixels

### Adaptive Thresholding (for shadow......)

- Divide the image into sub-images
- Compute thresholds for each sub-image
- Interpolate thresholds for every point using bilinear interpolation

### Band Thresholding

- Could be used for edge detection

## Mathematical Morphology

- Based on algebra of non-linear operators operating on object shape
- Performs many tasks better and more quickly than standard approaches
- Separate part of image analysis
- Main uses:
  - Pre-processing
  - Object structure enhancement
  - Segmentation
  - Description of objects

### Dilation and Erosion

### Closing and Opening

## Connectivity

- Search image row by row
  - Label each non-zero pixel
  - Assign new label if previous pixels are background
  - Else pick any label from previous pixels
  - Note equivalence if any of other pixels have a different label

- Relabel equivalent Labels

```cpp
#include<opencv2/core/core.hpp>
#include<opencv2/highgui/highgui.hpp>
#include<opencv2/imgproc.hpp>
#include<iostream>
#include<vector>

using namespace std;
using namespace cv;

int main() {
	string path1 = "../data/girl.png";
	string path2 = "../data/bin.png";
	string path3 = "../data/airplane.png";

	Mat img = imread(path1, IMREAD_COLOR);

	vector<vector<Point>> contours;
	vector<Vec4i> hierarchy;
	Mat gray, binaryImg, contoursImg;
	cvtColor(img, gray, COLOR_RGB2GRAY);

	threshold(gray, binaryImg, 180, 255, THRESH_BINARY);
	findContours(binaryImg, contours, hierarchy, RETR_TREE, CHAIN_APPROX_NONE);

	imshow("original img", img);

	for (int contour = 0; contour < contours.size(); contour++) {
		Scalar colour(rand() & 0xFF, rand() & 0xFF, rand() & 0xFF);
		drawContours(img, contours, contour, contour, FILLED, 8, hierarchy);
	}

	imshow("Contour Imgae", img);

	waitKey(0);
	return 0;
}
```

![]({{ '/pic/contour.png' | relative_url }})