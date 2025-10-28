---
layout: default
title: Smoothing and Noise
nav_order: 3
permalink: /smoothing-and-noise/
description: Median and Gaussian blurs, HSV thresholding, and basic denoising in OpenCV.
---

```cpp
#include<opencv2/opencv.hpp>
#include<iostream>

using namespace std;
using namespace cv;

const int max_value_H = 360 / 2;
const int max_value = 255;

int main() {
	string path1 = "../data/airplane.png";
	string path2 = "../data/dog.png";
	string path3 = "../data/girl.png";

	
	Mat img1 = imread(path1, IMREAD_COLOR);
	resize(img1, img1, { 500, 500 });

	if (img1.empty()) {
		cout << "Could not read the image" << endl;
		return 1;
	}
	vector<int> lower_bound = { 170, 80, 55 };

	int low_H = lower_bound[0], low_S = lower_bound[1], low_V = lower_bound[2];
	int high_H = max_value_H, high_S = max_value, high_V = max_value;

	Mat hsvImg, imgThreshold;

	// convert from BGR to HSV
	cvtColor(img1, hsvImg, COLOR_BGR2HSV);
	// detect the object based on HSV range values
	inRange(hsvImg, Scalar(low_H, low_S, low_V), Scalar(high_H, high_S, high_V), imgThreshold);

	Mat medianBlurImg, gaussianBlurImg;
	medianBlur(img1, medianBlurImg, 9);
	GaussianBlur(img1, gaussianBlurImg, Size(1, 1), 9, 9);


	imshow("img1", img1);
	imshow("median", medianBlurImg);
	imshow("gaussian", gaussianBlurImg);
	//imshow("img1", img1);
	imshow("hsv", hsvImg);
	imshow("threshold", imgThreshold);
	waitKey(0);
	return 0;
}
```

